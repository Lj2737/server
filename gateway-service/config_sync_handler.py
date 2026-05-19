"""
智能胸牌服务管理系统 - 词库配置同步请求处理器
核心功能：
1. 接收后端主动调用，同步词库配置到所有健康的算力节点
2. 入参校验：sop、forbidden、customer至少一类不能为空
3. 调用NodeManager的广播方法，将配置同步到所有健康的算力节点
4. 等待所有节点返回同步结果
5. 所有节点同步成功返回success=true，部分失败返回success=false并标注失败节点
6. 完整日志记录：配置版本号、同步时间、节点同步状态、失败原因

对应后端接口：POST /badge/v1/algorithm/config/sync
算力节点内部接口：POST /badge/v1/internal/algorithm/config/sync

处理流程（严格按顺序）：
    步骤1：入参校验，sop、forbidden、customer至少一类不能为空
    步骤2：生成配置版本号（时间戳格式），调用广播方法
    步骤3：等待所有节点返回同步结果
    步骤4：统计成功/失败节点，构建返回结果
    步骤5：记录同步日志：配置版本号、同步时间、节点同步状态、失败原因

请求体格式（后端→网关）：
    {
        "sop": [],
        "forbidden": [],
        "customer": []
    }

返回体格式（网关→后端）：
    全部成功：
    {
        "code": 200,
        "msg": "ok",
        "data": {
            "success": true,
            "configVersion": "20260508103200",
            "successCount": 4,
            "failCount": 0,
            "details": [...]
        }
    }

    部分失败：
    {
        "code": 200,
        "msg": "ok",
        "data": {
            "success": false,
            "configVersion": "20260508103200",
            "successCount": 3,
            "failCount": 1,
            "details": [...]
        }
    }

使用示例：
    from config_sync_handler import ConfigSyncHandler
    from node_manager import NodeManager

    # 获取单例（由main.py在启动时初始化）
    handler = ConfigSyncHandler()

    # 处理词库同步请求
    result = await handler.handle_config_sync(request_body={
        "sop": [],
        "forbidden": [],
        "customer": [],
    })
"""
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from config import (
    CONFIG_SYNC_INTERNAL_PATH,
    CONFIG_SYNC_REQUEST_TIMEOUT,
    ConfigType,
    REQUEST_ID_HEADER,
    DATETIME_FORMAT,
)
from node_manager import NodeManager
from http_client import HttpClientSingleton
from exception import ErrorCode, build_error_response
from utils import TimeFormatter


class ConfigSyncHandler:
    """
    词库配置同步请求处理器 - 单例模式

    职责：
    - 接收后端主动调用的词库配置同步请求
    - 校验sop/forbidden/customer至少一类非空
    - 广播配置到所有健康的算力节点
    - 等待所有节点返回同步结果
    - 统计成功/失败节点，构建返回结果

    设计约束：
    - 必须和NodeManager联动，实现配置广播
    - 全异步实现，并发向所有健康节点发送同步请求
    - 单节点失败不影响其他节点的同步
    - 日志必须完整记录全链路交互
    """

    _instance: Optional["ConfigSyncHandler"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个ConfigSyncHandler实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._node_manager: Optional[NodeManager] = None
        self._initialized = False

    def initialize(self, node_manager: NodeManager) -> None:
        """
        初始化词库配置同步处理器
        在FastAPI lifespan startup阶段调用

        Args:
            node_manager: 节点管理器实例（已启动健康检查）
        """
        if self._initialized:
            return
        self._node_manager = node_manager
        self._initialized = True
        logger.info(
            f"词库配置同步处理器初始化完成 | "
            f"内部路径={CONFIG_SYNC_INTERNAL_PATH} | "
            f"超时={CONFIG_SYNC_REQUEST_TIMEOUT}s"
        )

    # ==================== 核心处理方法 ====================

    async def handle_config_sync(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理词库配置同步请求
        完整实现步骤1~5，由backend_router调用

        Args:
            request_body: 后端发送的请求体
                {
                    "sop": [],
                    "forbidden": [],
                    "customer": []
                }

        Returns:
            标准响应体：
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "success": true/false,
                    "configVersion": "20260508103200",
                    "successCount": 4,
                    "failCount": 0,
                    "details": [...]
                },
                "request_id": "..."
            }
        """
        if not self._initialized or self._node_manager is None:
            logger.error("词库配置同步处理器未初始化，拒绝请求")
            return build_error_response(
                code=ErrorCode.INTERNAL_ERROR,
                msg="配置同步服务未就绪",
            )

        request_id = str(uuid.uuid4())

        # ========== 步骤1：入参校验 ==========
        validation_error = self._validate_request(request_body, request_id)
        if validation_error:
            return validation_error

        sop = request_body.get("sop", [])
        forbidden = request_body.get("forbidden", [])
        customer = request_body.get("customer", [])

        # ========== 步骤2：生成配置版本号，准备广播 ==========
        # 配置版本号格式：yyyyMMddHHmmss
        config_version = TimeFormatter.now().replace("-", "").replace(":", "").replace(" ", "")

        logger.info(
            f"词库配置同步请求开始 | request_id={request_id} | "
            f"configVersion={config_version} | "
            f"sop={len(sop)} | forbidden={len(forbidden)} | customer={len(customer)}"
        )

        # 获取所有健康节点
        healthy_nodes = [
            addr
            for addr, node in self._node_manager._nodes.items()
            if node.is_healthy
        ]

        if not healthy_nodes:
            logger.warning(
                f"词库配置同步失败 | 无可用健康节点 | request_id={request_id}"
            )
            return {
                "code": 200,
                "msg": "ok",
                "data": {
                    "success": False,
                    "configVersion": config_version,
                    "successCount": 0,
                    "failCount": 0,
                    "details": [],
                    "error": "无可用健康节点",
                },
                "request_id": request_id,
            }

        # ========== 步骤3：并发向所有健康节点发送同步请求 ==========
        tasks = [
            self._sync_config_to_node(addr, request_body, config_version, request_id)
            for addr in healthy_nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ========== 步骤4：统计成功/失败节点 ==========
        success_count = 0
        fail_count = 0
        details: List[Dict[str, Any]] = []

        for addr, result in zip(healthy_nodes, results):
            if isinstance(result, Exception):
                # 单节点同步失败
                fail_count += 1
                error_msg = str(result)[:200]
                details.append({
                    "node": addr,
                    "status": "failed",
                    "error": error_msg,
                })
                logger.error(
                    f"词库配置同步失败 | request_id={request_id} | "
                    f"节点={addr} | 错误={error_msg}"
                )
            else:
                # 单节点同步成功
                success_count += 1
                # 更新节点配置版本号
                await self._node_manager.update_node_config_version(addr, config_version)
                details.append({
                    "node": addr,
                    "status": "success",
                })
                logger.info(
                    f"词库配置同步成功 | request_id={request_id} | "
                    f"节点={addr} | 版本={config_version}"
                )

        # ========== 步骤5：构建返回结果 ==========
        all_success = fail_count == 0

        sync_result = {
            "success": all_success,
            "configVersion": config_version,
            "successCount": success_count,
            "failCount": fail_count,
            "details": details,
        }

        logger.info(
            f"词库配置同步完成 | request_id={request_id} | "
            f"configVersion={config_version} | "
            f"成功={success_count} | 失败={fail_count} | "
            f"overall={all_success}"
        )

        return {
            "code": 200,
            "msg": "ok",
            "data": sync_result,
            "request_id": request_id,
        }

    # ==================== 入参校验 ====================

    @staticmethod
    def _validate_request(
        request_body: Dict[str, Any], request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        校验词库配置同步请求的必传字段

        必传字段：
        - sop/forbidden/customer: 三类词库配置，至少一类非空

        Args:
            request_body: 请求体
            request_id: 请求ID

        Returns:
            None: 校验通过
            Dict: 校验失败的错误响应
        """
        for group_name in ("sop", "forbidden", "customer"):
            group_value = request_body.get(group_name, [])
            if group_value is None:
                request_body[group_name] = []
                continue
            if not isinstance(group_value, list):
                logger.warning(
                    f"词库同步入参校验失败 | {group_name}非数组 | request_id={request_id}"
                )
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"{group_name}必须为数组",
                    request_id=request_id,
                )

        if (
            not request_body.get("sop")
            and not request_body.get("forbidden")
            and not request_body.get("customer")
        ):
            logger.warning(
                f"词库同步入参校验失败 | sop/forbidden/customer均为空 | request_id={request_id}"
            )
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="sop、forbidden、customer至少有一类不能为空",
                request_id=request_id,
            )

        return None  # 校验通过

    # ==================== 单节点同步 ====================

    async def _sync_config_to_node(
        self,
        address: str,
        request_body: Dict[str, Any],
        config_version: str,
        request_id: str,
    ) -> dict:
        """
        向单个算力节点同步词库配置
        请求路径：POST http://{address}/badge/v1/internal/algorithm/config/sync

        重要：算力节点Pydantic模型使用snake_case字段名，
        此处必须发送 snake_case 元字段（config_type/config_version）和文档分组（sop/forbidden/customer），
        不能发旧版 camelCase 格式（configType/configVersion/configData），
        否则算力节点参数校验会返回422失败。

        字段映射（后端camelCase → 算力节点snake_case）：
        - 固定补充 config_type=KEYWORD
        - （网关生成）→ config_version
        - sop/forbidden/customer → 原样透传

        Args:
            address: 节点地址
            request_body: 原始请求体（后端camelCase格式）
            config_version: 配置版本号（网关生成）
            request_id: 请求ID

        Returns:
            节点响应内容

        Raises:
            Exception: 同步失败时抛出异常
        """
        client = await HttpClientSingleton.get_client()
        url = f"http://{address}{CONFIG_SYNC_INTERNAL_PATH}"

        # 构建同步请求体：网关补充内部元字段，三类词库数据按文档原样透传。
        sync_body = {
            "config_type": ConfigType.KEYWORD,
            "config_version": config_version,
            "sop": request_body.get("sop", []),
            "forbidden": request_body.get("forbidden", []),
            "customer": request_body.get("customer", []),
        }

        forward_headers = {
            REQUEST_ID_HEADER: request_id,
            "Content-Type": "application/json",
        }

        logger.debug(
            f"词库配置同步发送 | request_id={request_id} | "
            f"节点={address} | 版本={config_version} | "
            f"sop={len(sync_body['sop'])} | forbidden={len(sync_body['forbidden'])} | "
            f"customer={len(sync_body['customer'])}"
        )

        response = await client.post(
            url=url,
            json=sync_body,
            headers=forward_headers,
            timeout=CONFIG_SYNC_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    # ==================== 状态查询 ====================

    def is_initialized(self) -> bool:
        """配置同步处理器是否已初始化"""
        return self._initialized
