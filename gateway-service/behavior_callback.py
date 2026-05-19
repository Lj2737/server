"""
智能胸牌服务管理系统 - 语音行为识别结果回调后端
核心功能：
1. 接收主网关从算力节点拿到的行为识别推理结果
2. 按v3.1文档要求的格式封装请求体（multipart/form-data）
3. 通过BackendClient主动回调后端接口
4. 异常行为片段录音合并到同一请求中上传（v3.1变更：不再单独调用录音上传接口）

v3.1变更说明：
- 接口路径：POST /badge/v1/internal/ai/voice-behaviors
- Content-Type：multipart/form-data（不再是application/json）
- 表单字段：metadata（JSON字符串）+ file（异常行为片段录音，ABNORMAL必传）
- 废弃：独立的录音上传接口 POST /badge/v1/internal/ai/recordings

对应v3.1文档接口：POST /badge/v1/internal/ai/voice-behaviors
Content-Type: multipart/form-data

处理流程（严格按顺序）：
    步骤1：接收算力节点返回的推理结果（data字段）
    步骤2：调用IdGenerator生成全局唯一eventId
    步骤3：调用TimeFormatter格式化eventTime
    步骤4：按文档封装metadata JSON（eventTime、deviceNo、behaviorType、summary、configItemId、keywordContent）
    步骤5：判断behaviorType是否为ABNORMAL，若是则附加异常音频文件
    步骤6：调用BackendClient发送multipart/form-data到后端
    步骤7：回调失败记录完整日志，不影响主网关运行

metadata格式（v3.1文档5.2节）：
    {
        "eventTime": "2026-05-08 10:31:00",
        "deviceNo": "BADGE0001",
        "behaviorType": "ABNORMAL",
        "summary": "员工触发服务禁语",
        "configItemId": "forbidden-service-attitude",
        "keywordContent": "你自己看"
    }

使用示例：
    from behavior_callback import BehaviorCallback
    from utils import BackendClient

    # 初始化
    callback = BehaviorCallback()
    callback.initialize(backend_client=BackendClient())

    # 触发回调
    import asyncio
    asyncio.create_task(callback.handle_result(
        inference_data={
            "behavior_type": "ABNORMAL",
            "summary": "员工触发服务禁语",
            "config_item_id": "forbidden-service-attitude",
            "keyword_content": "你自己看",
            "is_abnormal": True,
            "abnormal_audio_clip": "UklGRi4AAABX...",
        },
        device_no="BADGE0001",
        event_time="2026-05-08 10:30:00",
    ))
"""
import base64
import json
from typing import Any, Dict, Optional

from loguru import logger

from config import BEHAVIOR_CALLBACK_PATH, BehaviorType
from utils import BackendClient, IdGenerator, TimeFormatter


class BehaviorCallback:
    """
    语音行为识别结果回调处理器 - 单例模式

    职责：
    - 接收算力节点返回的行为识别推理结果
    - 封装成v3.1文档要求的multipart/form-data格式
    - 主动回调后端 POST /badge/v1/internal/ai/voice-behaviors
    - 异常行为片段录音合并到同一请求中上传

    设计约束：
    - 全异步实现，不阻塞主网关路由转发
    - 使用BackendClient保证鉴权/重试/日志规则统一
    - 使用IdGenerator保证eventId全局唯一和幂等
    - 使用TimeFormatter保证时间格式统一
    - 回调失败仅记日志，不影响主流程
    """

    _instance: Optional["BehaviorCallback"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个BehaviorCallback实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._backend_client: Optional[BackendClient] = None
        self._initialized = False

    def initialize(self, backend_client: BackendClient) -> None:
        """
        初始化回调处理器
        在FastAPI lifespan startup阶段调用

        Args:
            backend_client: 后端通用客户端实例（已初始化）
        """
        if self._initialized:
            return
        self._backend_client = backend_client
        self._initialized = True
        logger.info(
            f"行为识别回调处理器初始化完成 | "
            f"回调路径={BEHAVIOR_CALLBACK_PATH} | "
            f"Content-Type=multipart/form-data（v3.1）"
        )

    # ==================== 核心回调方法 ====================

    async def handle_result(
        self,
        inference_data: Dict[str, Any],
        device_no: str,
        event_time: str = "",
    ) -> None:
        """
        处理行为识别推理结果，回调后端
        完整实现步骤1~7，由router通过asyncio.create_task异步调用

        v3.1变更：异常行为片段录音合并到voice-behaviors请求中
        - ABNORMAL行为：metadata + file（异常音频片段）
        - STANDARD/CUSTOMER行为：仅metadata，无file

        Args:
            inference_data: 算力节点返回的data字段
                {
                    "behavior_type": "ABNORMAL",
                    "summary": "员工未按SOP回应顾客问题",
                    "is_abnormal": true,
                    "abnormal_audio_clip": "UklGRi4AAABX..."  # 可选，仅ABNORMAL时
                }
            device_no: 设备编号（从原始请求的device_no字段提取）
            event_time: 行为发生时间（从原始请求的event_time字段提取）
        """
        if not self._initialized or self._backend_client is None:
            logger.error("行为识别回调处理器未初始化，跳过回调")
            return

        # ========== 步骤2：调用IdGenerator生成全局唯一eventId ==========
        event_id = IdGenerator.generate_behavior_id()

        # ========== 步骤3：调用TimeFormatter格式化eventTime ==========
        formatted_time = self._format_event_time(event_time)

        # ========== 步骤4：按文档封装metadata JSON ==========
        behavior_type = self._validate_behavior_type(
            inference_data.get("behavior_type", BehaviorType.STANDARD)
        )
        summary = inference_data.get("summary", "")
        config_item_id = self._first_non_empty(
            inference_data.get("configItemId"),
            inference_data.get("config_item_id"),
        )
        keyword_content = self._first_non_empty(
            inference_data.get("keywordContent"),
            inference_data.get("keyword_content"),
        )

        metadata = {
            "eventTime": formatted_time,
            "deviceNo": device_no,
            "behaviorType": behavior_type,
            "summary": summary,
            "configItemId": config_item_id,
            "keywordContent": keyword_content,
        }
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        logger.info(
            f"行为识别回调开始 | eventId={event_id} | "
            f"deviceNo={device_no} | behaviorType={behavior_type} | "
            f"eventTime={formatted_time} | configItemId={config_item_id}"
        )

        # ========== 步骤5：判断是否为ABNORMAL，附加异常音频文件 ==========
        files = None
        abnormal_audio_clip = inference_data.get("abnormal_audio_clip")

        if behavior_type == BehaviorType.ABNORMAL and abnormal_audio_clip:
            try:
                # base64解码异常音频片段
                audio_bytes = base64.b64decode(abnormal_audio_clip)
                files = {
                    "file": (
                        f"{event_id}.wav",   # 文件名
                        audio_bytes,          # 文件字节
                        "audio/wav",          # Content-Type
                    ),
                }
                logger.info(
                    f"异常行为音频片段已附加 | eventId={event_id} | "
                    f"音频大小={len(audio_bytes) / 1024:.1f}KB"
                )
            except Exception as e:
                logger.error(
                    f"异常音频base64解码失败 | eventId={event_id} | "
                    f"错误类型={type(e).__name__} | 错误={str(e)[:200]}"
                )
                # ABNORMAL但没有有效音频，仍发送metadata（由后端判断）
                # v3.1文档：ABNORMAL必须上传file，否则后端返回"异常行为必须上传语音内容"
                # 这里我们尽力发送，如果解码失败记录日志但继续发送

        # ========== 步骤6：调用BackendClient发送multipart/form-data到后端 ==========
        data = {
            "metadata": metadata_json,
        }

        try:
            if files:
                # ABNORMAL：metadata + file
                result = await self._backend_client.post_multipart(
                    path=BEHAVIOR_CALLBACK_PATH,
                    files=files,
                    data=data,
                    idempotency_key=event_id,
                )
            else:
                # STANDARD/CUSTOMER：仅metadata（使用multipart发送，但无file）
                # v3.1文档：STANDARD和CUSTOMER不上传file
                # 但接口统一使用multipart/form-data
                result = await self._backend_client.post_multipart(
                    path=BEHAVIOR_CALLBACK_PATH,
                    files=None,
                    data=data,
                    idempotency_key=event_id,
                )
            logger.info(
                f"行为识别回调成功 | eventId={event_id} | "
                f"deviceNo={device_no} | behaviorType={behavior_type} | "
                f"后端响应={result}"
            )
        except Exception as e:
            # ========== 步骤7：回调失败记录完整日志，不影响主流程 ==========
            logger.error(
                f"行为识别回调失败 | eventId={event_id} | "
                f"deviceNo={device_no} | behaviorType={behavior_type} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:300]}"
            )

    # ==================== 辅助方法 ====================

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        """返回第一个非空字符串值，用于兼容算力节点snake_case和文档camelCase字段。"""
        for value in values:
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
        return ""

    @staticmethod
    def _format_event_time(event_time: str) -> str:
        """
        格式化事件时间
        - 优先使用原始请求中的event_time（硬件上报时间）
        - 通过TimeFormatter.parse()校验格式合法性
        - 格式不合法时回退为TimeFormatter.now()当前时间

        Args:
            event_time: 原始事件时间字符串

        Returns:
            格式化后的事件时间字符串
        """
        if event_time:
            parsed = TimeFormatter.parse(event_time)
            if parsed:
                return TimeFormatter.format_datetime(parsed)

        # 回退为当前时间
        fallback = TimeFormatter.now()
        if event_time:
            logger.warning(
                f"原始event_time格式不合法，回退为当前时间 | "
                f"原始值={event_time} | 回退值={fallback}"
            )
        return fallback

    @staticmethod
    def _validate_behavior_type(behavior_type: str) -> str:
        """
        校验behavior_type枚举值
        - 必须使用v3.1文档规定的全大写枚举值
        - 非标准值回退为STANDARD并记录警告日志

        Args:
            behavior_type: 行为类型字符串

        Returns:
            合法的枚举值字符串
        """
        valid_types = [BehaviorType.STANDARD, BehaviorType.ABNORMAL, BehaviorType.CUSTOMER]
        if behavior_type not in valid_types:
            logger.warning(
                f"behavior_type枚举值异常，回退为STANDARD | "
                f"原始值={behavior_type} | 允许值={valid_types}"
            )
            return BehaviorType.STANDARD
        return behavior_type

    # ==================== 状态查询 ====================

    def is_initialized(self) -> bool:
        """回调处理器是否已初始化"""
        return self._initialized
