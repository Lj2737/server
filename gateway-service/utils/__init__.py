"""
智能胸牌服务管理系统 - 后端对接通用工具类包
统一管理所有和后端交互相关的工具类，方便后期维护和扩展

包含4个核心工具：
1. BackendClient  - 后端接口通用客户端（鉴权、重试、日志）
2. IdGenerator    - 幂等ID生成器（eventId）
3. TimeFormatter  - 时间格式化工具（yyyy-MM-dd HH:mm:ss）
4. AudioTempStorage - 异常音频临时存储（base64→WAV，定时清理）

使用方式：
    # 方式一：从utils包直接导入（推荐）
    from utils import BackendClient, IdGenerator, TimeFormatter, AudioTempStorage

    # 方式二：从子模块导入
    from utils.backend_client import BackendClient
    from utils.id_generator import IdGenerator
    from utils.time_formatter import TimeFormatter
    from utils.audio_temp_storage import AudioTempStorage
"""

from utils.backend_client import BackendClient
from utils.id_generator import IdGenerator
from utils.time_formatter import TimeFormatter
from utils.audio_temp_storage import AudioTempStorage

__all__ = [
    "BackendClient",
    "IdGenerator",
    "TimeFormatter",
    "AudioTempStorage",
]
