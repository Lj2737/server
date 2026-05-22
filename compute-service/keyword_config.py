"""
智能胸牌服务管理系统 - 词库配置管理
核心功能：
1. 接收主网关广播的词库配置，本地缓存到内存+JSON文件
2. 服务重启时自动从本地JSON文件加载配置
3. 版本号校验：低版本不重复处理
4. 更新LLM推理模块的词库引用，实时生效，无需重启服务
"""
import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from loguru import logger

from config import KEYWORD_CONFIG_FILE, ConfigType


class KeywordConfigManager:
    """
    词库配置管理器
    - 内存缓存 + 本地JSON文件双存储
    - 重启自动从JSON文件恢复配置
    - 版本校验防止重复处理
    - 实时更新LLM推理模块的词库引用
    """

    def __init__(self):
        """初始化词库配置管理器"""
        # 当前生效的配置版本号
        self._current_version: str = ""
        # 当前生效的词库配置内容（内存缓存）
        self._current_config: Dict[str, Any] = {}
        # 词库配置项按sop/forbidden/customer分组缓存（供LLM推理引用）
        self._keyword_groups: Dict[str, List[Dict[str, Any]]] = {
            "sop": [],
            "forbidden": [],
            "customer": [],
        }

        # 确保数据目录存在
        config_path = Path(KEYWORD_CONFIG_FILE)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 启动时从本地JSON文件恢复配置
        self._load_from_file()

    def _load_from_file(self) -> None:
        """从本地JSON文件加载词库配置（重启恢复）"""
        if os.path.exists(KEYWORD_CONFIG_FILE):
            try:
                with open(KEYWORD_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._current_version = data.get("config_version", "")
                self._current_config = data.get("config_data", {})
                self._keyword_groups = self._normalize_config_data(self._current_config)

                logger.info(
                    f"词库配置从本地文件恢复 | 版本={self._current_version} | "
                    f"条目数={self._count_keywords(self._keyword_groups)}"
                )
            except Exception as e:
                logger.error(f"词库配置文件加载失败 | 路径={KEYWORD_CONFIG_FILE} | 错误={e}")
                # 加载失败时使用空配置，不影响服务启动
                self._current_version = ""
                self._current_config = {}
                self._keyword_groups = {"sop": [], "forbidden": [], "customer": []}
        else:
            logger.info("本地词库配置文件不存在，使用空配置")

    def _save_to_file(self) -> None:
        """将当前词库配置保存到本地JSON文件"""
        try:
            data = {
                "config_version": self._current_version,
                "config_data": self._current_config,
            }
            with open(KEYWORD_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"词库配置已保存到本地文件 | 版本={self._current_version} | "
                f"路径={KEYWORD_CONFIG_FILE}"
            )
        except Exception as e:
            logger.error(f"词库配置文件保存失败 | 错误={e}")

    def sync_config(
        self,
        config_type: str,
        config_version: str,
        config_data: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """
        同步词库配置（主网关广播调用）
        - 校验config_type必须为KEYWORD
        - 版本号校验：低版本不重复处理
        - 更新内存缓存 + 本地文件
        - 实时更新LLM推理模块的词库引用

        Args:
            config_type: 配置类型，必须为KEYWORD
            config_version: 配置版本号
            config_data: 按sop/forbidden/customer分组的配置数据

        Returns:
            同步是否成功
        """
        # 校验配置类型
        if config_type != ConfigType.KEYWORD:
            logger.warning(f"配置类型不匹配 | 期望=KEYWORD | 实际={config_type}")
            return False

        # 版本号校验：传入版本低于当前生效版本，跳过处理
        if self._current_version and config_version <= self._current_version:
            logger.info(
                f"词库配置版本低于或等于当前版本，跳过同步 | "
                f"当前版本={self._current_version} | 传入版本={config_version}"
            )
            return True  # 返回成功，避免主网关重试

        # 更新内存缓存
        old_version = self._current_version
        self._current_version = config_version
        self._keyword_groups = self._normalize_config_data(config_data)
        self._current_config = {
            "config_type": config_type,
            "config_version": config_version,
            **self._keyword_groups,
        }

        # 保存到本地JSON文件
        self._save_to_file()

        logger.info(
            f"词库配置同步完成 | 旧版本={old_version} → 新版本={config_version} | "
            f"条目数={self._count_keywords(self._keyword_groups)}"
        )
        return True

    def get_current_version(self) -> str:
        """获取当前生效的配置版本号"""
        return self._current_version

    def get_keyword_text(self) -> str:
        """
        获取词库配置的文本表示（供LLM推理使用）
        将词库配置序列化为文本，注入到LLM的user_prompt中

        Returns:
            词库配置文本，空配置时返回"暂无词库配置"
        """
        if not self._keyword_groups or self._count_keywords(self._keyword_groups) == 0:
            return "暂无词库配置"

        parts = []
        group_titles = {
            "sop": "STANDARD/SOP话术",
            "forbidden": "ABNORMAL/违禁词",
            "customer": "CUSTOMER/顾客关键词",
        }
        for group_key in ("sop", "forbidden", "customer"):
            config_items = self._keyword_groups.get(group_key, [])
            if not config_items:
                continue
            parts.append(f"[{group_titles[group_key]}]")
            for config_item in config_items:
                config_item_id = config_item.get("configItemId", "")
                config_item_name = config_item.get("configItemName", "")
                parts.append(f"- 配置项: {config_item_name} ({config_item_id})")
                for keyword in config_item.get("keywords", []):
                    content = keyword.get("content", "")
                    keyword_id = keyword.get("id", "")
                    match_type = keyword.get("matchType")
                    if content:
                        parts.append(
                            f"  - {content} | keywordId={keyword_id} | matchType={match_type}"
                        )

        return "\n".join(parts) if parts else "暂无词库配置"

    def get_keyword_items(self) -> List[Dict[str, Any]]:
        """获取当前词库配置条目列表（兼容旧调用，返回三类配置项的扁平列表）"""
        items: List[Dict[str, Any]] = []
        for group_key, config_items in self._keyword_groups.items():
            for item in config_items:
                flattened = dict(item)
                flattened["configGroup"] = group_key
                items.append(flattened)
        return items

    def get_keyword_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取当前按sop/forbidden/customer分组的词库配置"""
        return self._keyword_groups

    def find_keyword_match(self, group_key: str, text: str) -> Optional[Dict[str, str]]:
        source_text = (text or "").strip()
        if not source_text:
            return None

        normalized_source = source_text.lower()
        for config_item in self._keyword_groups.get(group_key, []):
            config_item_id = str(config_item.get("configItemId", "")).strip()
            if not config_item_id:
                continue
            for keyword in config_item.get("keywords", []) or []:
                content = str(keyword.get("content", "")).strip()
                if content and content.lower() in normalized_source:
                    return {
                        "config_item_id": config_item_id,
                        "keyword_content": content,
                    }
        return None

    @staticmethod
    def _normalize_config_data(config_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """归一化配置数据，仅保留文档约定的三类顶层字段。"""
        if "items" in config_data and not any(
            key in config_data for key in ("sop", "forbidden", "customer")
        ):
            legacy_items = list(config_data.get("items") or [])
            return {
                "sop": legacy_items,
                "forbidden": [],
                "customer": [],
            }
        return {
            "sop": list(config_data.get("sop") or []),
            "forbidden": list(config_data.get("forbidden") or []),
            "customer": list(config_data.get("customer") or []),
        }

    @staticmethod
    def _count_keywords(config_data: Dict[str, List[Dict[str, Any]]]) -> int:
        """统计三类配置下的关键词总数。"""
        total = 0
        for config_items in config_data.values():
            for config_item in config_items:
                total += len(config_item.get("keywords") or [])
        return total
