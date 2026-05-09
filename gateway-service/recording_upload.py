"""
智能胸牌服务管理系统 - 异常行为片段录音上传后端
核心功能：
1. 仅当behaviorType=ABNORMAL时触发，将异常行为片段录音上传到后端
2. 触发时机：行为识别回调成功，且is_abnormal=true时触发
3. 对应接口：POST /internal/badge/ai/recordings，Content-Type: multipart/form-data

对应v3文档接口：POST /internal/badge/ai/recordings

处理流程（严格按顺序）：
    步骤1：接收行为识别回调模块传递的eventId（behaviorEventId）和abnormal_audio_clip（base64）
    步骤2：调用IdGenerator生成录音上传的全局唯一eventId
    步骤3：调用AudioTempStorage将base64解码成临时WAV文件
    步骤4：封装multipart/form-data表单，仅包含2个必传字段：
           - file：临时WAV文件
           - metadata：JSON字符串，包含eventId和behaviorEventId
    步骤5：调用BackendClient通用客户端，发送POST请求到后端接口
    步骤6：上传成功后，删除临时WAV文件
    步骤7：上传失败时，记录完整错误日志，保留临时文件，支持手动重试

metadata固定格式：
    {
        "eventId": "AI_RECORDING_20260508103200001",
        "behaviorEventId": "上一步行为回调的eventId"
    }

强制规则：
    - 仅上传异常行为片段，禁止上传全量录音、标准行为录音
    - 全异步实现，不阻塞主网关的路由转发
    - 必须使用通用工具类：BackendClient、IdGenerator、AudioTempStorage

使用示例：
    from recording_upload import RecordingUpload
    from utils import BackendClient, AudioTempStorage

    # 初始化（在FastAPI lifespan startup中调用，由main.py统一管理）
    upload = RecordingUpload()
    await upload.initialize(
        backend_client=BackendClient(),
        audio_storage=AudioTempStorage(),
    )

    # 注册为BehaviorCallback的音频处理器
    from behavior_callback import BehaviorCallback
    callback = BehaviorCallback()
    callback.register_audio_clip_handler(upload.handle_audio_clip)

    # 当行为识别结果为ABNORMAL时，BehaviorCallback会自动调用：
    await upload.handle_audio_clip(
        behavior_event_id="AI_BEHAVIOR_20260508103100_001234",
        abnormal_audio_clip="UklGRi4AAABXQVZFZm10...",
    )
"""
import json
import os
from typing import Optional

from loguru import logger

from config import RECORDING_UPLOAD_PATH, RECORDING_UPLOAD_TIMEOUT
from utils import BackendClient, IdGenerator, AudioTempStorage


class RecordingUpload:
    """
    异常行为片段录音上传处理器 - 单例模式

    职责：
    - 接收BehaviorCallback传递的behavior_event_id和abnormal_audio_clip
    - 生成录音上传的eventId，保存临时WAV文件
    - 封装multipart/form-data上传到后端 POST /internal/badge/ai/recordings
    - 上传成功删除临时文件，失败保留文件支持重试

    设计约束：
    - 全异步实现，不阻塞主网关路由转发
    - 必须使用通用工具类：BackendClient、IdGenerator、AudioTempStorage
    - 日志完整记录：录音eventId、behaviorEventId、文件大小、上传状态、错误信息
    - 所有配置项抽离到config.py
    """

    _instance: Optional["RecordingUpload"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个RecordingUpload实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._backend_client: Optional[BackendClient] = None
        self._audio_storage: Optional[AudioTempStorage] = None
        self._initialized = False

    async def initialize(
        self,
        backend_client: BackendClient,
        audio_storage: AudioTempStorage,
    ) -> None:
        """
        初始化录音上传处理器
        在FastAPI lifespan startup阶段调用

        Args:
            backend_client: 后端通用客户端实例（已初始化，用于发送multipart请求）
            audio_storage: 音频临时存储实例（已初始化，用于保存/读取/删除临时WAV文件）
        """
        if self._initialized:
            return
        self._backend_client = backend_client
        self._audio_storage = audio_storage
        self._initialized = True
        logger.info(
            f"录音上传处理器初始化完成 | "
            f"上传路径={RECORDING_UPLOAD_PATH} | "
            f"超时={RECORDING_UPLOAD_TIMEOUT}s"
        )

    # ==================== 核心上传方法 ====================

    async def handle_audio_clip(
        self,
        behavior_event_id: str,
        abnormal_audio_clip: str,
    ) -> None:
        """
        处理异常音频片段，上传到后端
        完整实现步骤1~7，由BehaviorCallback在回调成功后调用

        此方法作为BehaviorCallback.register_audio_clip_handler()的处理器，
        仅当behaviorType=ABNORMAL且包含abnormal_audio_clip时才会被调用。

        Args:
            behavior_event_id: 行为识别回调的eventId（步骤1传入）
                格式：AI_BEHAVIOR_yyyyMMddHHmmss_6位随机序号
            abnormal_audio_clip: 异常音频片段的base64编码数据（步骤1传入）
                由算力节点从原始录音中截取的异常行为片段
        """
        if not self._initialized or self._backend_client is None or self._audio_storage is None:
            logger.error("录音上传处理器未初始化，跳过上传")
            return

        # ========== 参数校验 ==========
        if not behavior_event_id:
            logger.error("录音上传参数异常 | behavior_event_id为空，跳过上传")
            return
        if not abnormal_audio_clip:
            logger.error("录音上传参数异常 | abnormal_audio_clip为空，跳过上传")
            return

        logger.info(
            f"异常行为片段录音上传开始 | behaviorEventId={behavior_event_id} | "
            f"音频数据长度={len(abnormal_audio_clip)}字符"
        )

        # ========== 步骤2：调用IdGenerator生成录音上传的全局唯一eventId ==========
        recording_event_id = IdGenerator.generate_recording_id()
        logger.debug(
            f"录音上传eventId已生成 | recordingEventId={recording_event_id} | "
            f"behaviorEventId={behavior_event_id}"
        )

        # ========== 步骤3：调用AudioTempStorage将base64解码成临时WAV文件 ==========
        temp_file_path: Optional[str] = None
        try:
            temp_file_path = await self._audio_storage.save(
                event_id=recording_event_id,
                behavior_event_id=behavior_event_id,
                base64_audio=abnormal_audio_clip,
            )
        except ValueError as e:
            # base64解码失败
            logger.error(
                f"录音上传失败（base64解码异常）| recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | 错误={str(e)[:300]}"
            )
            return
        except IOError as e:
            # 文件写入失败
            logger.error(
                f"录音上传失败（临时文件写入异常）| recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | 错误={str(e)[:300]}"
            )
            return
        except Exception as e:
            logger.error(
                f"录音上传失败（临时文件保存未知异常）| recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:300]}"
            )
            return

        # ========== 步骤4~5：读取文件、封装表单、上传 ==========
        try:
            # 读取临时WAV文件字节
            file_bytes = await self._audio_storage.read(temp_file_path)
            file_size_kb = len(file_bytes) / 1024

            logger.info(
                f"录音文件准备上传 | recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | "
                f"文件大小={file_size_kb:.1f}KB | "
                f"临时文件={os.path.basename(temp_file_path)}"
            )

            # 封装metadata JSON字符串
            metadata = {
                "eventId": recording_event_id,
                "behaviorEventId": behavior_event_id,
            }
            metadata_json = json.dumps(metadata, ensure_ascii=False)

            # 封装multipart/form-data表单
            # 字段1：file - WAV文件
            # 字段2：metadata - JSON字符串
            files = {
                "file": (
                    f"{recording_event_id}.wav",  # 文件名
                    file_bytes,                    # 文件字节
                    "audio/wav",                   # Content-Type
                ),
            }
            data = {
                "metadata": metadata_json,
            }

            # 调用BackendClient发送POST multipart请求到后端
            result = await self._backend_client.post_multipart(
                path=RECORDING_UPLOAD_PATH,
                files=files,
                data=data,
                idempotency_key=recording_event_id,  # 重试时保持同一eventId
                timeout=RECORDING_UPLOAD_TIMEOUT,
            )

            logger.info(
                f"异常行为片段录音上传成功 | recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | "
                f"文件大小={file_size_kb:.1f}KB | 后端响应={result}"
            )

            # ========== 步骤6：上传成功后，删除临时WAV文件 ==========
            await self._audio_storage.delete(temp_file_path)
            logger.debug(
                f"临时音频文件已删除（上传成功）| recordingEventId={recording_event_id} | "
                f"文件={os.path.basename(temp_file_path)}"
            )

        except Exception as e:
            # ========== 步骤7：上传失败时，记录完整错误日志，保留临时文件 ==========
            # 保留临时文件，不删除，以便手动重试
            file_size_info = ""
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    file_size_kb = os.path.getsize(temp_file_path) / 1024
                    file_size_info = f" | 临时文件保留={os.path.basename(temp_file_path)} | 文件大小={file_size_kb:.1f}KB"
                except Exception:
                    file_size_info = f" | 临时文件保留={os.path.basename(temp_file_path)}"

            logger.error(
                f"异常行为片段录音上传失败 | recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:300]}{file_size_info}"
            )
            # 不re-raise，不影响BehaviorCallback主流程

    # ==================== 手动重试方法 ====================

    async def retry_upload(self, temp_file_path: str, behavior_event_id: str) -> None:
        """
        手动重试上传
        当上传失败后，临时文件被保留，可通过此方法手动重试

        Args:
            temp_file_path: 临时WAV文件路径（由上次上传失败时保留）
            behavior_event_id: 关联的行为事件ID

        使用场景：
            上传失败后，运维人员排查问题后，使用此方法重新上传
            无需重新走行为识别流程
        """
        if not self._initialized or self._backend_client is None or self._audio_storage is None:
            logger.error("录音上传处理器未初始化，跳过重试上传")
            return

        if not os.path.exists(temp_file_path):
            logger.error(f"重试上传失败 | 临时文件不存在 | 路径={temp_file_path}")
            return

        # 从临时文件名中提取recording_event_id
        filename = os.path.basename(temp_file_path)
        recording_event_id = filename.replace(".wav", "")

        logger.info(
            f"手动重试上传开始 | recordingEventId={recording_event_id} | "
            f"behaviorEventId={behavior_event_id} | 文件={filename}"
        )

        try:
            # 读取临时文件
            file_bytes = await self._audio_storage.read(temp_file_path)
            file_size_kb = len(file_bytes) / 1024

            # 封装metadata和表单
            metadata = {
                "eventId": recording_event_id,
                "behaviorEventId": behavior_event_id,
            }
            metadata_json = json.dumps(metadata, ensure_ascii=False)

            files = {
                "file": (
                    filename,
                    file_bytes,
                    "audio/wav",
                ),
            }
            data = {
                "metadata": metadata_json,
            }

            result = await self._backend_client.post_multipart(
                path=RECORDING_UPLOAD_PATH,
                files=files,
                data=data,
                idempotency_key=recording_event_id,
                timeout=RECORDING_UPLOAD_TIMEOUT,
            )

            logger.info(
                f"手动重试上传成功 | recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | "
                f"文件大小={file_size_kb:.1f}KB | 后端响应={result}"
            )

            # 重试成功后删除临时文件
            await self._audio_storage.delete(temp_file_path)
            logger.debug(
                f"临时音频文件已删除（重试上传成功）| recordingEventId={recording_event_id} | "
                f"文件={filename}"
            )

        except Exception as e:
            logger.error(
                f"手动重试上传失败 | recordingEventId={recording_event_id} | "
                f"behaviorEventId={behavior_event_id} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:300]} | "
                f"临时文件保留={temp_file_path}"
            )

    # ==================== 状态查询 ====================

    def is_initialized(self) -> bool:
        """录音上传处理器是否已初始化"""
        return self._initialized
