"""
智能胸牌服务管理系统 - 幂等ID生成器
核心功能：
1. 按固定规则生成eventId，格式：AI_业务类型_yyyyMMddHHmmss_6位随机序号
2. 业务类型枚举：BEHAVIOR（行为识别）、RECORDING（录音上传）
3. 生成的ID全局唯一，可作为幂等键

格式示例：
- AI_BEHAVIOR_20260508103100_001234
- AI_RECORDING_20260508103200_005678

严格对齐v3文档：
- 5.2 语音行为识别回调：eventId = "AI_BEHAVIOR_20260508103100001"
- 5.3 异常行为片段录音上传：eventId = "AI_RECORDING_20260508103200001"

使用示例：
    from utils import IdGenerator  # 或 from utils.id_generator import IdGenerator
    from config import BusinessType

    # 生成行为识别事件ID
    event_id = IdGenerator.generate(BusinessType.BEHAVIOR)
    # 输出：AI_BEHAVIOR_20260508143522_382741

    # 生成录音上传事件ID
    recording_id = IdGenerator.generate(BusinessType.RECORDING)
    # 输出：AI_RECORDING_20260508143522_729184
"""
import random
import datetime
from typing import Optional

from loguru import logger

from config import BusinessType


class IdGenerator:
    """
    幂等ID生成器
    - 固定格式：AI_业务类型_yyyyMMddHHmmss_6位随机序号
    - 全局唯一：时间戳精确到秒 + 6位随机数（百万分之一碰撞概率）
    - 可用作算法回调后端的幂等键（eventId）
    - 无状态，纯函数生成，不需要初始化
    """

    @staticmethod
    def generate(business_type: str, dt: Optional[datetime.datetime] = None) -> str:
        """
        生成全局唯一的幂等事件ID

        格式规则：AI_{业务类型}_{yyyyMMddHHmmss}_{6位随机序号}
        - 业务类型：BEHAVIOR / RECORDING
        - 时间部分：精确到秒，14位数字
        - 随机序号：6位数字，范围000000-999999

        Args:
            business_type: 业务类型枚举值（BusinessType.BEHAVIOR / BusinessType.RECORDING）
            dt: 可选的指定时间（默认使用当前时间，主要用于测试）

        Returns:
            全局唯一的事件ID字符串

        Raises:
            ValueError: 业务类型不在允许的枚举范围内
        """
        # 校验业务类型
        valid_types = [BusinessType.BEHAVIOR, BusinessType.RECORDING]
        if business_type not in valid_types:
            raise ValueError(
                f"业务类型无效: {business_type}，"
                f"允许值: {valid_types}"
            )

        # 生成时间戳部分：yyyyMMddHHmmss
        if dt is None:
            dt = datetime.datetime.now()
        timestamp_part = dt.strftime("%Y%m%d%H%M%S")

        # 生成6位随机序号：000000-999999
        random_part = f"{random.randint(0, 999999):06d}"

        # 组装eventId
        event_id = f"AI_{business_type}_{timestamp_part}_{random_part}"

        logger.debug(f"生成幂等ID | 类型={business_type} | ID={event_id}")
        return event_id

    @staticmethod
    def generate_behavior_id(dt: Optional[datetime.datetime] = None) -> str:
        """
        便捷方法：生成行为识别事件ID
        等价于 IdGenerator.generate(BusinessType.BEHAVIOR)

        Args:
            dt: 可选的指定时间

        Returns:
            行为识别事件ID，如 AI_BEHAVIOR_20260508103100_001234
        """
        return IdGenerator.generate(BusinessType.BEHAVIOR, dt)

    @staticmethod
    def generate_recording_id(dt: Optional[datetime.datetime] = None) -> str:
        """
        便捷方法：生成录音上传事件ID
        等价于 IdGenerator.generate(BusinessType.RECORDING)

        Args:
            dt: 可选的指定时间

        Returns:
            录音上传事件ID，如 AI_RECORDING_20260508103200_005678
        """
        return IdGenerator.generate(BusinessType.RECORDING, dt)

    @staticmethod
    def parse(event_id: str) -> Optional[dict]:
        """
        解析事件ID，提取业务类型、时间戳和随机序号
        主要用于日志排查和调试

        Args:
            event_id: 事件ID字符串

        Returns:
            解析结果字典，格式无效时返回None
            {
                "prefix": "AI",
                "business_type": "BEHAVIOR",
                "timestamp": "20260508103100",
                "sequence": "001234",
                "datetime": datetime(2026, 5, 8, 10, 31, 0),
            }
        """
        try:
            parts = event_id.split("_")
            if len(parts) != 4 or parts[0] != "AI":
                return None

            prefix, business_type, timestamp_str, sequence = parts

            # 解析时间戳
            dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

            return {
                "prefix": prefix,
                "business_type": business_type,
                "timestamp": timestamp_str,
                "sequence": sequence,
                "datetime": dt,
            }
        except (ValueError, IndexError):
            return None
