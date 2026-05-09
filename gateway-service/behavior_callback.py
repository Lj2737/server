"""
智能胸牌服务管理系统 - 语音行为识别结果回调后端
核心功能：
1. 接收主网关从算力节点拿到的行为识别推理结果
2. 按v3文档要求的格式封装请求体
3. 通过BackendClient主动回调后端接口
4. 异常音频片段传递预留（对接后续录音上传模块）

对应v3文档接口：POST /internal/badge/ai/voice-behaviors
Content-Type: application/json

处理流程（严格按顺序）：
    步骤1：接收算力节点返回的推理结果（data字段）
    步骤2：调用IdGenerator生成全局唯一eventId
    步骤3：调用TimeFormatter格式化eventTime
    步骤4：按v3文档封装请求体（仅保留必传字段，禁止冗余）
    步骤5：调用BackendClient发送POST到后端
    步骤6：回调成功后传递eventId和abnormal_audio_clip给录音上传模块（预留）
    步骤7：回调失败记录完整日志，不影响主网关运行

请求体固定格式：
    {
        "eventId": "AI_BEHAVIOR_20260508103100_001234",
        "eventTime": "2026-05-08 10:31:00",
        "deviceNo": "BADGE0001",
        "behaviorType": "STANDARD / ABNORMAL / CUSTOMER",
        "summary": "行为摘要文本"
    }

使用示例：
    from behavior_callback import BehaviorCallback
    from utils import BackendClient

    # 初始化（在FastAPI lifespan startup中调用）
    callback = BehaviorCallback()
    callback.initialize(backend_client=BackendClient())

    # 触发回调（由router在推理成功后调用，fire-and-forget）
    import asyncio
    asyncio.create_task(callback.handle_result(
        inference_data={
            "behavior_type": "ABNORMAL",
            "summary": "员工未按SOP回应顾客问题",
            "is_abnormal": True,
            "abnormal_audio_clip": "UklGRi4AAABX...",
        },
        device_no="BADGE0001",
        event_time="2026-05-08 10:30:00",
    ))

    # 注册异常音频片段处理器（对接录音上传模块时使用）
    async def my_audio_handler(event_id: str, audio_clip: str):
        # 录音上传模块的具体实现
        pass
    callback.register_audio_clip_handler(my_audio_handler)
"""
import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from loguru import logger

from config import BEHAVIOR_CALLBACK_PATH, BehaviorType
from utils import BackendClient, IdGenerator, TimeFormatter


class BehaviorCallback:
    """
    语音行为识别结果回调处理器 - 单例模式

    职责：
    - 接收算力节点返回的行为识别推理结果
    - 封装成v3文档要求的格式
    - 主动回调后端 POST /internal/badge/ai/voice-behaviors
    - 异常音频片段传递预留（register_audio_clip_handler）

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
        # 异常音频片段处理器（预留，对接录音上传模块）
        self._audio_clip_handler: Optional[Callable[[str, str], Awaitable[None]]] = None
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
            f"回调路径={BEHAVIOR_CALLBACK_PATH}"
        )

    def register_audio_clip_handler(
        self, handler: Callable[[str, str], Awaitable[None]]
    ) -> None:
        """
        注册异常音频片段处理器
        用于对接后续的录音上传模块，当行为识别结果为ABNORMAL且包含音频片段时调用

        Args:
            handler: 异步回调函数
                     参数1: event_id (str) - 行为识别的eventId
                     参数2: abnormal_audio_clip (str) - 异常音频Base64数据

        示例：
            async def handle_audio_clip(event_id: str, audio_clip: str):
                # 调用录音上传模块
                await upload_recording(event_id, audio_clip)

            callback.register_audio_clip_handler(handle_audio_clip)
        """
        self._audio_clip_handler = handler
        logger.info("异常音频片段处理器已注册")

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

        Args:
            inference_data: 算力节点返回的data字段
                {
                    "behavior_type": "ABNORMAL",
                    "summary": "员工未按SOP回应顾客问题",
                    "is_abnormal": true,
                    "abnormal_audio_clip": "UklGRi4AAABX..."  # 可选，仅ABNORMAL时
                }
            device_no: 设备编号（从原始请求的device_no字段提取）
            event_time: 行为发生时间（从原始请求的event_time字段提取，
                        格式yyyy-MM-dd HH:mm:ss，为空时使用当前时间）
        """
        if not self._initialized or self._backend_client is None:
            logger.error("行为识别回调处理器未初始化，跳过回调")
            return

        # ========== 步骤2：调用IdGenerator生成全局唯一eventId ==========
        event_id = IdGenerator.generate_behavior_id()

        # ========== 步骤3：调用TimeFormatter格式化eventTime ==========
        formatted_time = self._format_event_time(event_time)

        # ========== 步骤4：按v3文档封装请求体（仅必传字段） ==========
        behavior_type = self._validate_behavior_type(
            inference_data.get("behavior_type", BehaviorType.STANDARD)
        )
        summary = inference_data.get("summary", "")

        callback_body = {
            "eventId": event_id,
            "eventTime": formatted_time,
            "deviceNo": device_no,
            "behaviorType": behavior_type,
            "summary": summary,
        }

        logger.info(
            f"行为识别回调开始 | eventId={event_id} | "
            f"deviceNo={device_no} | behaviorType={behavior_type} | "
            f"eventTime={formatted_time}"
        )

        # ========== 步骤5：调用BackendClient发送POST到后端 ==========
        try:
            result = await self._backend_client.post(
                path=BEHAVIOR_CALLBACK_PATH,
                json_body=callback_body,
                idempotency_key=event_id,  # 重试时保持同一eventId
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
            return  # 不re-raise，不影响主网关运行

        # ========== 步骤6：回调成功后传递abnormal_audio_clip给录音上传模块 ==========
        abnormal_audio_clip = inference_data.get("abnormal_audio_clip")
        if (
            behavior_type == BehaviorType.ABNORMAL
            and abnormal_audio_clip
            and self._audio_clip_handler
        ):
            try:
                await self._audio_clip_handler(event_id, abnormal_audio_clip)
                logger.info(
                    f"异常音频片段已传递给录音上传模块 | "
                    f"eventId={event_id} | 音频数据长度={len(abnormal_audio_clip)}字符"
                )
            except Exception as e:
                logger.error(
                    f"异常音频片段传递失败 | eventId={event_id} | "
                    f"错误类型={type(e).__name__} | 错误={str(e)[:300]}"
                )
        elif behavior_type == BehaviorType.ABNORMAL and abnormal_audio_clip:
            # 有音频片段但没有注册处理器，记录提示
            logger.debug(
                f"异常音频片段未传递（未注册处理器）| eventId={event_id}"
            )

    # ==================== 辅助方法 ====================

    @staticmethod
    def _format_event_time(event_time: str) -> str:
        """
        格式化事件时间
        - 优先使用原始请求中的event_time（硬件上报时间）
        - 通过TimeFormatter.parse()校验格式合法性
        - 格式不合法时回退为TimeFormatter.now()当前时间

        Args:
            event_time: 原始事件时间字符串，格式yyyy-MM-dd HH:mm:ss

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
        - 必须使用v3文档规定的全大写枚举值
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

    def has_audio_clip_handler(self) -> bool:
        """是否已注册异常音频片段处理器"""
        return self._audio_clip_handler is not None
