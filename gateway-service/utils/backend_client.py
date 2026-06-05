"""
智能胸牌服务管理系统 - 后端接口通用客户端
核心功能：
1. 封装统一的异步HTTP客户端（基于httpx.AsyncClient），单例模式，复用连接池
2. 内置鉴权Header：支持Header Token鉴权，Token可配置，预留签名鉴权扩展位
3. 内置重试机制：最多3次，指数退避1s/3s/5s，仅对5xx/超时/连接失败重试，4xx不重试
4. 重试时保持同一个业务幂等ID，禁止生成新ID
5. 内置超时控制：默认10秒，可配置
6. 内置完整日志记录：请求URL、方法、请求体、响应状态码、响应体、耗时、错误信息

使用示例：
    from utils import BackendClient  # 或 from utils.backend_client import BackendClient

    # 初始化（在FastAPI lifespan中调用）
    client = BackendClient()
    await client.initialize()

    # POST请求 - 回调后端语音行为
    result = await client.post(
        path="/badge/v1/internal/ai/voice-behaviors",
        json_body={
            "eventId": "AI_BEHAVIOR_20260508103100001",
            "eventTime": "2026-05-08 10:31:00",
            "deviceNo": "BADGE0001",
            "behaviorType": "ABNORMAL",
            "summary": "员工未按SOP回应顾客问题",
        },
        idempotency_key="AI_BEHAVIOR_20260508103100001",
    )

    # GET请求 - 查询后端数据
    result = await client.get(
        path="/badge/v1/internal/ai/devices/knowledge-base",
    )

    # 关闭（在FastAPI shutdown中调用）
    await client.close()
"""
import asyncio
import inspect
import time
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
from loguru import logger

from config import (
    BACKEND_BASE_URL,
    BACKEND_AUTH_TOKEN,
    BACKEND_AUTH_HEADER,
    BACKEND_REQUEST_TIMEOUT,
    BACKEND_MAX_RETRIES,
    BACKEND_RETRY_DELAYS,
    BACKEND_HTTP_MAX_CONNECTIONS,
    BACKEND_HTTP_MAX_KEEPALIVE_CONNECTIONS,
    DEVICE_EVENT_FORWARD_PATH,
    DEVICE_EVENT_FORWARD_MAX_RETRIES,
    DEVICE_EVENT_FORWARD_RETRY_INTERVAL,
    DIALOG_COMPLETION_CALLBACK_PATH,
    KNOWLEDGE_BASE_QUERY_PATH,
)

BADGE_BINDING_NOT_FOUND_CODE = 1022100006


class BackendClient:
    """
    后端接口通用客户端 - 单例模式
    所有和后端的HTTP交互都通过该客户端，保证规则统一：
    - 统一鉴权、统一超时、统一重试、统一日志
    - 复用TCP连接池，减少建连开销
    - 适配树莓派资源，控制连接池大小
    """

    _instance: Optional["BackendClient"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个BackendClient实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化后端客户端（仅首次创建时执行）
        - 创建httpx.AsyncClient实例
        - 配置鉴权Header
        - 不在__init__中发起任何网络请求
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False
        self._badge_binding_missing_handler: Optional[
            Callable[[str, str, str], Awaitable[None]]
        ] = None
        self._recent_device_event_keys: Dict[tuple, float] = {}

    def set_badge_binding_missing_handler(
        self,
        handler: Callable[[str, str, str], Awaitable[None]],
    ) -> None:
        """注册胸牌绑定缺失错误的异步处理器。"""
        self._badge_binding_missing_handler = handler

    async def initialize(self) -> None:
        """
        初始化httpx异步客户端
        在FastAPI lifespan startup阶段调用
        - 创建连接池
        - 预建鉴权Header
        """
        if self._initialized and self._client and not self._client.is_closed:
            return

        # 构建鉴权Header（预留签名鉴权扩展位）
        default_headers = self._build_auth_headers()

        self._client = httpx.AsyncClient(
            base_url=BACKEND_BASE_URL,
            headers=default_headers,
            timeout=httpx.Timeout(
                connect=5.0,                      # 建立TCP连接超时5秒
                read=BACKEND_REQUEST_TIMEOUT,      # 读取响应超时
                write=10.0,                        # 写入请求体超时10秒
                pool=5.0,                          # 从连接池获取连接超时5秒
            ),
            limits=httpx.Limits(
                max_connections=BACKEND_HTTP_MAX_CONNECTIONS,
                max_keepalive_connections=BACKEND_HTTP_MAX_KEEPALIVE_CONNECTIONS,
            ),
            follow_redirects=False,  # 禁止重定向
        )
        self._initialized = True
        logger.info(
            f"后端客户端初始化完成 | 基础地址={BACKEND_BASE_URL} | "
            f"最大重试={BACKEND_MAX_RETRIES} | 超时={BACKEND_REQUEST_TIMEOUT}s"
        )

    def _build_auth_headers(self) -> Dict[str, str]:
        """
        构建鉴权Header
        当前实现：Header Token鉴权
        预留扩展：签名鉴权可在此方法中扩展，不影响调用方

        Returns:
            鉴权Header字典
        """
        headers = {}
        if BACKEND_AUTH_TOKEN:
            # Token格式：Bearer {token} 或直接 {token}，取决于后端约定
            headers[BACKEND_AUTH_HEADER] = f"Bearer {BACKEND_AUTH_TOKEN}"
        return headers

    async def close(self) -> None:
        """
        关闭httpx客户端，释放连接池资源
        在FastAPI lifespan shutdown阶段调用
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info("后端客户端已关闭，连接池资源已释放")

    # ==================== 核心请求方法 ====================

    async def post(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        发送POST请求到后端

        Args:
            path: 接口路径（相对于BACKEND_BASE_URL），如 /badge/v1/internal/ai/voice-behaviors
            json_body: 请求体JSON数据
            idempotency_key: 幂等键（如eventId），重试时保持同一ID
            extra_headers: 额外的请求头（会覆盖默认鉴权头）
            timeout: 本次请求超时时间（秒），不传则使用默认值

        Returns:
            后端响应的JSON数据

        Raises:
            httpx.HTTPStatusError: 后端返回4xx错误（不重试，直接抛出）
            RuntimeError: 重试耗尽后仍然失败
        """
        return await self._request(
            method="POST",
            path=path,
            json_body=json_body,
            idempotency_key=idempotency_key,
            extra_headers=extra_headers,
            timeout=timeout,
        )

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        发送GET请求到后端

        Args:
            path: 接口路径
            params: 查询参数
            extra_headers: 额外的请求头
            timeout: 本次请求超时时间（秒）

        Returns:
            后端响应的JSON数据
        """
        return await self._request(
            method="GET",
            path=path,
            params=params,
            extra_headers=extra_headers,
            timeout=timeout,
        )

    async def post_multipart(
        self,
        path: str,
        files: Optional[Dict[str, tuple]],
        data: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        发送multipart/form-data请求到后端
        用于上传录音文件等场景

        Args:
            path: 接口路径
            files: 文件字段，格式 {"field_name": ("filename", bytes, "content_type")}
            data: 表单文本字段
            idempotency_key: 幂等键
            extra_headers: 额外的请求头
            timeout: 超时时间

        Returns:
            后端响应的JSON数据
        """
        if not files and data:
            files = {
                field_name: (None, "" if field_value is None else str(field_value))
                for field_name, field_value in data.items()
            }
            data = None

        return await self._request(
            method="POST",
            path=path,
            files=files,
            data=data,
            idempotency_key=idempotency_key,
            extra_headers=extra_headers,
            timeout=timeout,
        )

    # ==================== 硬件状态透传方法 ====================

    async def forward_device_event(self, event_data: dict) -> bool:
        """
        透传硬件状态到后端（异步，不阻塞主事件循环）

        核心原则（强制透传规则，违反会导致后端状态错乱）：
        - 不能修改reportTime，必须保留硬件的原始上报时间
        - 不能新增任何字段（比如算法的receiveTime、requestId）
        - 不能删除任何字段（哪怕后端暂时不用）
        - 不能修改payload内部的任何字段和值

        重试机制：
        - 最多重试2次，间隔1秒
        - 仅对5xx错误、网络超时、连接失败重试
        - 4xx错误不重试
        - 重试时保持原始event_data完全不变，不生成新的ID

        Args:
            event_data: 硬件上报的原始数据字典，包含deviceNo/eventType/reportTime/payload

        Returns:
            True: 转发成功（后端返回2xx）
            False: 转发失败（重试耗尽、4xx错误、客户端未初始化）
        """
        event_data = self._normalize_device_event_for_backend(event_data)

        # 前置检查：客户端是否可用
        if not self._initialized or self._client is None or self._client.is_closed:
            logger.error(
                f"硬件状态转发失败 | 原因=后端客户端未初始化或已关闭 | "
                f"原始数据={event_data}"
            )
            return False

        if self._is_duplicate_device_event(event_data):
            logger.info(
                "Skip duplicate device heartbeat forward | "
                f"deviceNo={event_data.get('deviceNo', 'unknown')} | "
                f"payload={event_data.get('payload', {})}"
            )
            return True

        # 提取关键业务字段用于日志
        device_no = event_data.get("deviceNo", "未知")
        event_type = event_data.get("eventType", "未知")
        report_time = event_data.get("reportTime", "未知")

        start_time = time.time()
        last_error: Optional[Exception] = None

        # 重试循环：1次首次 + N次重试
        for attempt in range(1 + DEVICE_EVENT_FORWARD_MAX_RETRIES):
            try:
                # 直接透传原始数据，不添加任何额外字段（无幂等键、无receiveTime）
                response = await self._client.post(
                    url=DEVICE_EVENT_FORWARD_PATH,
                    json=event_data,
                )

                elapsed_ms = int((time.time() - start_time) * 1000)

                # 2xx 成功
                if 200 <= response.status_code < 300:
                    logger.info(
                        f"硬件状态转发后端成功 | "
                        f"deviceNo={device_no} | eventType={event_type} | "
                        f"reportTime={report_time} | 耗时={elapsed_ms}ms"
                    )
                    return True

                # 4xx 客户端错误 - 不重试
                elif 400 <= response.status_code < 500:
                    error_body = response.text[:500]
                    logger.error(
                        f"硬件状态转发后端客户端错误(不重试) | "
                        f"deviceNo={device_no} | eventType={event_type} | "
                        f"reportTime={report_time} | "
                        f"状态码={response.status_code} | 响应={error_body} | "
                        f"耗时={elapsed_ms}ms | 原始数据={event_data}"
                    )
                    return False

                # 5xx 服务端错误 - 触发重试
                else:
                    error_body = response.text[:500]
                    last_error = httpx.HTTPStatusError(
                        f"后端服务端错误: HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    logger.warning(
                        f"硬件状态转发后端服务端错误(将重试) | "
                        f"deviceNo={device_no} | eventType={event_type} | "
                        f"reportTime={report_time} | "
                        f"状态码={response.status_code} | 响应={error_body} | "
                        f"重试次数={attempt + 1}/{DEVICE_EVENT_FORWARD_MAX_RETRIES} | "
                        f"耗时={elapsed_ms}ms"
                    )

            except httpx.TimeoutException as e:
                # 请求超时 - 触发重试
                elapsed_ms = int((time.time() - start_time) * 1000)
                last_error = e
                logger.warning(
                    f"硬件状态转发后端超时(将重试) | "
                    f"deviceNo={device_no} | eventType={event_type} | "
                    f"reportTime={report_time} | "
                    f"超时类型={type(e).__name__} | "
                    f"重试次数={attempt + 1}/{DEVICE_EVENT_FORWARD_MAX_RETRIES} | "
                    f"耗时={elapsed_ms}ms"
                )

            except httpx.ConnectError as e:
                # 连接失败 - 触发重试
                elapsed_ms = int((time.time() - start_time) * 1000)
                last_error = e
                logger.warning(
                    f"硬件状态转发后端连接失败(将重试) | "
                    f"deviceNo={device_no} | eventType={event_type} | "
                    f"reportTime={report_time} | "
                    f"错误={str(e)[:200]} | "
                    f"重试次数={attempt + 1}/{DEVICE_EVENT_FORWARD_MAX_RETRIES} | "
                    f"耗时={elapsed_ms}ms"
                )

            except Exception as e:
                # 其他未知异常 - 触发重试
                elapsed_ms = int((time.time() - start_time) * 1000)
                last_error = e
                logger.warning(
                    f"硬件状态转发后端异常(将重试) | "
                    f"deviceNo={device_no} | eventType={event_type} | "
                    f"reportTime={report_time} | "
                    f"异常={type(e).__name__}: {str(e)[:200]} | "
                    f"重试次数={attempt + 1}/{DEVICE_EVENT_FORWARD_MAX_RETRIES} | "
                    f"耗时={elapsed_ms}ms"
                )

            # 重试前等待（固定间隔），最后一次失败不需要等待
            if attempt < DEVICE_EVENT_FORWARD_MAX_RETRIES:
                logger.info(
                    f"硬件状态转发等待重试 | "
                    f"deviceNo={device_no} | 等待={DEVICE_EVENT_FORWARD_RETRY_INTERVAL}s"
                )
                await asyncio.sleep(DEVICE_EVENT_FORWARD_RETRY_INTERVAL)

        # 所有重试均失败
        total_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"硬件状态转发后端失败(重试耗尽) | "
            f"deviceNo={device_no} | eventType={event_type} | "
            f"reportTime={report_time} | "
            f"重试次数={DEVICE_EVENT_FORWARD_MAX_RETRIES} | 总耗时={total_ms}ms | "
            f"最后错误={type(last_error).__name__ if last_error else '未知'}: "
            f"{str(last_error)[:200] if last_error else ''} | "
            f"完整原始数据={event_data}"
        )
        return False

    @staticmethod
    def _normalize_device_event_for_backend(event_data: dict) -> dict:
        normalized = dict(event_data or {})
        payload = dict(normalized.get("payload") or {})
        normalized["payload"] = payload

        if normalized.get("eventType") != "HEARTBEAT":
            return normalized

        normalized["reportTime"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if "signalPercent" not in payload and "signalLevel" in payload:
            try:
                payload["signalPercent"] = int(payload["signalLevel"]) * 20
            except (TypeError, ValueError):
                payload["signalPercent"] = payload["signalLevel"]
        payload.pop("signalLevel", None)
        normalized["payload"] = payload
        return normalized

    def _is_duplicate_device_event(
        self,
        event_data: dict,
        window_seconds: float = 2.0,
    ) -> bool:
        if event_data.get("eventType") != "HEARTBEAT":
            return False

        payload = event_data.get("payload") or {}
        key = (
            event_data.get("deviceNo"),
            event_data.get("eventType"),
            payload.get("batteryLevel"),
            payload.get("signalPercent"),
        )
        if not key[0]:
            return False

        now = time.time()
        self._recent_device_event_keys = {
            cached_key: cached_time
            for cached_key, cached_time in self._recent_device_event_keys.items()
            if now - cached_time <= window_seconds
        }

        last_seen = self._recent_device_event_keys.get(key)
        if last_seen is not None and now - last_seen <= window_seconds:
            return True

        self._recent_device_event_keys[key] = now
        return False

    # ==================== 业务便捷方法 ====================

    async def report_dialog_completion(self, device_no: str, dialog_time: str) -> bool:
        """
        回调后端：AI对话完成通知

        POST /badge/v1/internal/ai/dialog-completions
        Content-Type: application/json
        Body: {"deviceNo": "BADGE0001", "dialogTime": "2026-05-08 10:35:00"}

        算力节点完成AI对话后，主网关调用此方法通知后端。
        后端收到后更新对话记录、触发后续业务流程。

        Args:
            device_no: 设备编号
            dialog_time: 对话完成时间，格式yyyy-MM-dd HH:mm:ss

        Returns:
            True: 回调成功（后端返回2xx）
            False: 回调失败（网络异常、后端返回非2xx、客户端未初始化）
        """
        try:
            result = await self.post(
                path=DIALOG_COMPLETION_CALLBACK_PATH,
                json_body={
                    "deviceNo": device_no,
                    "dialogTime": dialog_time,
                },
            )
            logger.info(
                f"AI对话完成回调成功 | deviceNo={device_no} | "
                f"dialogTime={dialog_time} | 后端响应={result}"
            )
            return True
        except Exception as e:
            logger.error(
                f"AI对话完成回调失败 | deviceNo={device_no} | "
                f"dialogTime={dialog_time} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:300]}"
            )
            return False

    async def get_knowledge_base_id(self, device_no: str) -> Optional[str]:
        """
        查询后端：获取设备对应的知识库ID


        POST /badge/v1/internal/ai/devices/knowledge-base
        Content-Type: application/json
        Body: {"deviceNo": "BADGE0001"}

        算力节点需要查询设备对应的知识库ID时，主网关调用此方法从后端获取。
        结果会被本地缓存（knowledge_base_cache），减少后端调用频次。

        Args:
            device_no: 设备编号

        Returns:
            knowledgeBaseId字符串，如 "dataset-001"
            None: 查询失败（网络异常、后端返回非2xx、客户端未初始化、响应中无knowledgeBaseId）
        """
        try:
            result = await self.post(
                path=KNOWLEDGE_BASE_QUERY_PATH,
                json_body={"deviceNo": device_no},
            )
            # 从后端响应中提取knowledgeBaseId
            knowledge_base_id = None
            if isinstance(result, dict):
                # 后端响应格式：{"code": 200, "msg": "ok", "data": {"knowledgeBaseId": "dataset-001"}}
                data = result.get("data", {})
                if isinstance(data, dict):
                    knowledge_base_id = data.get("knowledgeBaseId")
                # 兼容：data可能直接是knowledgeBaseId
                elif isinstance(data, str):
                    knowledge_base_id = data
                # 兼容文档返回：{"knowledgeBaseId": "dataset-001"}
                if knowledge_base_id is None:
                    knowledge_base_id = result.get("knowledgeBaseId")

            logger.info(
                f"知识库ID查询成功 | deviceNo={device_no} | "
                f"knowledgeBaseId={knowledge_base_id}"
            )
            return knowledge_base_id
        except Exception as e:
            logger.error(
                f"知识库ID查询失败 | deviceNo={device_no} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:300]}"
            )
            return None

    # ==================== 统一请求核心实现 ====================

    async def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, tuple]] = None,
        data: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        统一请求核心实现
        - 构建请求参数（URL、Header、Body）
        - 执行重试机制（仅5xx/超时/连接失败重试，4xx不重试）
        - 重试时保持同一个业务幂等ID
        - 记录完整请求/响应日志

        Args:
            method: HTTP方法（GET/POST等）
            path: 接口路径
            json_body: JSON请求体
            params: 查询参数
            files: 上传文件
            data: 表单数据
            idempotency_key: 幂等键
            extra_headers: 额外Header
            timeout: 超时时间

        Returns:
            后端响应JSON数据

        Raises:
            RuntimeError: 重试耗尽后仍然失败
        """
        if not self._initialized or self._client is None or self._client.is_closed:
            raise RuntimeError("后端客户端未初始化或已关闭，请先调用initialize()")

        # 合并Header：默认鉴权头 + 幂等键 + 额外头
        headers = {}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        if extra_headers:
            headers.update(extra_headers)

        # 构建请求超时
        req_timeout = timeout or BACKEND_REQUEST_TIMEOUT

        # 构建请求URL
        url = path

        # 记录请求开始
        start_time = time.time()
        request_desc = (
            f"方法={method} | 路径={url} | "
            f"幂等键={idempotency_key or '无'} | "
            f"请求体大小={len(str(json_body)) if json_body else 0}"
        )

        # 重试机制
        last_error: Optional[Exception] = None
        for attempt in range(1 + BACKEND_MAX_RETRIES):
            try:
                # 构建请求参数
                request_kwargs = {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "timeout": req_timeout,
                }

                # 根据请求类型添加参数
                if json_body is not None:
                    request_kwargs["json"] = json_body
                if params is not None:
                    request_kwargs["params"] = params
                if files is not None:
                    request_kwargs["files"] = files
                if data is not None:
                    request_kwargs["data"] = data

                # 发送请求
                response = await self._client.request(**request_kwargs)

                # 计算耗时
                elapsed_ms = int((time.time() - start_time) * 1000)

                # 检查响应状态码
                if 200 <= response.status_code < 300:
                    # 请求成功
                    response_data = response.json()
                    logger.info(
                        f"后端请求成功 | {request_desc} | "
                        f"状态码={response.status_code} | 耗时={elapsed_ms}ms"
                    )
                    return response_data

                elif 400 <= response.status_code < 500:
                    # 4xx客户端错误 - 不重试，直接抛出
                    error_body = response.text[:500]  # 截断避免日志过长
                    logger.warning(
                        f"后端请求客户端错误(不重试) | {request_desc} | "
                        f"状态码={response.status_code} | "
                        f"响应={error_body} | 耗时={elapsed_ms}ms"
                    )
                    self._dispatch_badge_binding_missing_broadcast(
                        response=response,
                        path=path,
                        json_body=json_body,
                        params=params,
                        data=data,
                    )
                    response.raise_for_status()

                else:
                    # 5xx服务端错误 - 触发重试
                    error_body = response.text[:500]
                    last_error = httpx.HTTPStatusError(
                        f"后端服务端错误: HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    logger.warning(
                        f"后端请求服务端错误(将重试) | {request_desc} | "
                        f"状态码={response.status_code} | "
                        f"响应={error_body} | "
                        f"重试次数={attempt + 1}/{BACKEND_MAX_RETRIES} | "
                        f"耗时={elapsed_ms}ms"
                    )

            except httpx.TimeoutException as e:
                # 请求超时 - 触发重试
                elapsed_ms = int((time.time() - start_time) * 1000)
                last_error = e
                logger.warning(
                    f"后端请求超时(将重试) | {request_desc} | "
                    f"超时类型={type(e).__name__} | "
                    f"重试次数={attempt + 1}/{BACKEND_MAX_RETRIES} | "
                    f"耗时={elapsed_ms}ms"
                )

            except httpx.ConnectError as e:
                # 连接失败 - 触发重试
                elapsed_ms = int((time.time() - start_time) * 1000)
                last_error = e
                logger.warning(
                    f"后端连接失败(将重试) | {request_desc} | "
                    f"错误={str(e)[:200]} | "
                    f"重试次数={attempt + 1}/{BACKEND_MAX_RETRIES} | "
                    f"耗时={elapsed_ms}ms"
                )

            except httpx.HTTPStatusError:
                # 4xx错误已在上方处理并re-raise，这里直接向上抛出
                raise

            except Exception as e:
                # 其他未知异常 - 触发重试
                elapsed_ms = int((time.time() - start_time) * 1000)
                last_error = e
                logger.warning(
                    f"后端请求异常(将重试) | {request_desc} | "
                    f"异常={type(e).__name__}: {str(e)[:200]} | "
                    f"重试次数={attempt + 1}/{BACKEND_MAX_RETRIES} | "
                    f"耗时={elapsed_ms}ms"
                )

            # 重试前等待（指数退避），最后一次失败不需要等待
            if attempt < BACKEND_MAX_RETRIES:
                delay = BACKEND_RETRY_DELAYS[attempt] if attempt < len(BACKEND_RETRY_DELAYS) else BACKEND_RETRY_DELAYS[-1]
                logger.info(f"后端请求等待重试 | 等待={delay}s | 幂等键={idempotency_key}")
                await asyncio.sleep(delay)

        # 所有重试均失败
        total_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"后端请求重试耗尽 | {request_desc} | "
            f"重试次数={BACKEND_MAX_RETRIES} | 总耗时={total_ms}ms | "
            f"最后错误={type(last_error).__name__ if last_error else '未知'}"
        )
        raise RuntimeError(
            f"后端请求失败，已重试{BACKEND_MAX_RETRIES}次 | "
            f"路径={path} | 最后错误={last_error}"
        )

    def _dispatch_badge_binding_missing_broadcast(
        self,
        response: httpx.Response,
        path: str,
        json_body: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, str]],
    ) -> None:
        payload = self._safe_response_json(response)
        if not self._is_badge_binding_missing_error(payload):
            return

        device_no = self._extract_device_no(json_body, params, data)
        if not device_no:
            logger.warning(
                f"胸牌绑定缺失播报跳过 | path={path} | 原因=deviceNo不存在"
            )
            return

        message = self._format_badge_binding_missing_broadcast(device_no, payload)
        handler = self._badge_binding_missing_handler
        if handler is None:
            logger.warning(
                f"胸牌绑定缺失播报跳过 | deviceNo={device_no} | "
                f"path={path} | 原因=未注册播报处理器 | 内容={message}"
            )
            return

        async def _run_broadcast() -> None:
            try:
                result = handler(device_no, message, path)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.error(
                    f"胸牌绑定缺失播报失败 | deviceNo={device_no} | "
                    f"path={path} | 错误类型={type(exc).__name__} | 错误={str(exc)[:300]}"
                )

        try:
            asyncio.create_task(_run_broadcast())
            logger.info(
                f"胸牌绑定缺失播报任务已创建 | deviceNo={device_no} | "
                f"path={path} | 内容={message}"
            )
        except RuntimeError as exc:
            logger.error(
                f"胸牌绑定缺失播报任务创建失败 | deviceNo={device_no} | "
                f"path={path} | 错误={exc}"
            )

    @staticmethod
    def _safe_response_json(response: httpx.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _is_badge_binding_missing_error(payload: Dict[str, Any]) -> bool:
        code = payload.get("code")
        msg = str(payload.get("msg") or payload.get("message") or "")
        return code == BADGE_BINDING_NOT_FOUND_CODE or "胸牌绑定记录不存在" in msg

    @staticmethod
    def _extract_device_no(
        json_body: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, str]],
    ) -> str:
        for source in (json_body, params, data):
            if not isinstance(source, dict):
                continue
            for key in ("deviceNo", "device_no", "deviceId", "device_id"):
                value = source.get(key)
                if value:
                    return str(value).strip()
        return ""

    @staticmethod
    def _format_badge_binding_missing_broadcast(
        device_no: str,
        payload: Dict[str, Any],
    ) -> str:
        raw_msg = str(payload.get("msg") or payload.get("message") or "胸牌绑定记录不存在")
        return f"设备{device_no}{raw_msg}，请检查后端胸牌绑定配置。"
