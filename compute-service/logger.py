"""
智能胸牌服务管理系统 - 算力节点日志配置
使用loguru实现控制台+文件双输出，按天分割，保留7天自动清理
适配树莓派资源，异步写入避免阻塞主线程
"""
import sys
from loguru import logger

from config import LOG_DIR, LOG_FILE_FORMAT, LOG_RETENTION_DAYS, LOG_ROTATION, LOG_LEVEL


def setup_logger() -> None:
    """
    初始化loguru日志配置
    - 移除默认handler，避免重复输出
    - 控制台输出：彩色格式，便于开发调试
    - 文件输出：按天分割，保留7天，异步写入
    """
    # 移除默认handler
    logger.remove()

    # 控制台输出 - 彩色格式
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=LOG_LEVEL,
        colorize=True,
    )

    # 文件输出 - 按天分割，保留7天
    logger.add(
        f"{LOG_DIR}/{LOG_FILE_FORMAT}",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,        # 每天0点轮转
        retention=LOG_RETENTION_DAYS,  # 保留7天
        compression="zip",             # 过期日志压缩
        encoding="utf-8",
        enqueue=True,                  # 异步写入（适配树莓派IO性能）
    )
