"""
智能胸牌服务管理系统 - 时间格式化工具
核心功能：
1. 统一输出yyyy-MM-dd HH:mm:ss格式的时间字符串
2. 支持将datetime对象、时间戳（秒/毫秒）转换为该格式
3. 获取当前时间的标准格式字符串
4. 所有和后端交互的时间字段都通过该工具格式化，禁止其他格式

严格对齐v3文档：
- 统一接口要求：时间格式联调前统一 yyyy-MM-dd HH:mm:ss
- 硬件上报时间 reportTime：2026-05-08 10:30:00
- 行为事件时间 eventTime：2026-05-08 10:31:00

使用示例：
    from utils import TimeFormatter  # 或 from utils.time_formatter import TimeFormatter

    # 获取当前时间（最常用）
    now_str = TimeFormatter.now()
    # 输出：2026-05-08 14:35:22

    # datetime对象转字符串
    import datetime
    dt = datetime.datetime(2026, 5, 8, 10, 30, 0)
    result = TimeFormatter.format_datetime(dt)
    # 输出：2026-05-08 10:30:00

    # 时间戳（秒）转字符串
    result = TimeFormatter.format_timestamp(1746678600)
    # 输出：2026-05-08 10:30:00

    # 时间戳（毫秒）转字符串
    result = TimeFormatter.format_timestamp_ms(1746678600000)
    # 输出：2026-05-08 10:30:00

    # 日期格式（yyyy-MM-dd）
    date_str = TimeFormatter.format_date(dt)
    # 输出：2026-05-08
"""
import time
import datetime
from typing import Optional

from loguru import logger

from config import DATETIME_FORMAT, DATE_FORMAT


class TimeFormatter:
    """
    时间格式化工具
    - 统一输出格式：yyyy-MM-dd HH:mm:ss（对齐v3文档要求）
    - 支持多种输入源：datetime对象、时间戳（秒/毫秒）
    - 无状态，纯函数，线程安全
    - 所有和后端交互的时间字段都必须通过此工具格式化
    """

    @staticmethod
    def now() -> str:
        """
        获取当前时间的标准格式字符串
        最常用的方法，行为识别、录音上传等场景直接调用

        Returns:
            当前时间字符串，格式：yyyy-MM-dd HH:mm:ss
            示例：2026-05-08 14:35:22
        """
        return datetime.datetime.now().strftime(DATETIME_FORMAT)

    @staticmethod
    def format_datetime(dt: datetime.datetime) -> str:
        """
        将datetime对象转换为标准格式字符串
        适用于已有datetime对象的场景

        Args:
            dt: datetime对象

        Returns:
            格式化后的时间字符串，如 2026-05-08 10:30:00
        """
        return dt.strftime(DATETIME_FORMAT)

    @staticmethod
    def format_date(dt: datetime.datetime) -> str:
        """
        将datetime对象转换为日期格式字符串（yyyy-MM-dd）
        适用于诊断总结的startDate/endDate字段

        Args:
            dt: datetime对象

        Returns:
            格式化后的日期字符串，如 2026-05-08
        """
        return dt.strftime(DATE_FORMAT)

    @staticmethod
    def format_timestamp(ts: float) -> str:
        """
        将Unix时间戳（秒）转换为标准格式字符串
        适用于从硬件或系统获取的时间戳

        Args:
            ts: Unix时间戳（秒级），如 1746678600.0

        Returns:
            格式化后的时间字符串，如 2026-05-08 10:30:00
        """
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.strftime(DATETIME_FORMAT)

    @staticmethod
    def format_timestamp_ms(ts_ms: float) -> str:
        """
        将Unix时间戳（毫秒）转换为标准格式字符串
        适用于Java后端返回的毫秒级时间戳

        Args:
            ts_ms: Unix时间戳（毫秒级），如 1746678600000.0

        Returns:
            格式化后的时间字符串，如 2026-05-08 10:30:00
        """
        ts_sec = ts_ms / 1000.0
        dt = datetime.datetime.fromtimestamp(ts_sec)
        return dt.strftime(DATETIME_FORMAT)

    @staticmethod
    def parse(time_str: str) -> Optional[datetime.datetime]:
        """
        将标准格式时间字符串解析为datetime对象
        适用于需要做时间计算、比较的场景

        Args:
            time_str: 时间字符串，格式必须是 yyyy-MM-dd HH:mm:ss

        Returns:
            datetime对象，解析失败返回None
        """
        try:
            return datetime.datetime.strptime(time_str, DATETIME_FORMAT)
        except ValueError:
            logger.warning(f"时间字符串解析失败 | 输入={time_str} | 期望格式=yyyy-MM-dd HH:mm:ss")
            return None

    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime.datetime]:
        """
        将日期字符串解析为datetime对象
        适用于诊断总结的startDate/EndDate字段

        Args:
            date_str: 日期字符串，格式必须是 yyyy-MM-dd

        Returns:
            datetime对象（时间部分为00:00:00），解析失败返回None
        """
        try:
            return datetime.datetime.strptime(date_str, DATE_FORMAT)
        except ValueError:
            logger.warning(f"日期字符串解析失败 | 输入={date_str} | 期望格式=yyyy-MM-dd")
            return None

    @staticmethod
    def today() -> str:
        """
        获取今天的日期字符串（yyyy-MM-dd）
        适用于诊断总结默认时间范围

        Returns:
            今天的日期字符串，如 2026-05-08
        """
        return datetime.datetime.now().strftime(DATE_FORMAT)

    @staticmethod
    def days_ago(days: int) -> str:
        """
        获取N天前的日期字符串（yyyy-MM-dd）
        适用于诊断总结的时间范围计算

        Args:
            days: 天数（正数表示过去）

        Returns:
            N天前的日期字符串，如 days=7 返回 2026-05-01
        """
        dt = datetime.datetime.now() - datetime.timedelta(days=days)
        return dt.strftime(DATE_FORMAT)
