"""
智能胸牌服务管理系统 - AI时段诊断总结请求处理器
核心功能：
1. 接收后端主动调用，校验入参
2. 生成requestId，通过NodeManager选择可用算力节点
3. 将后端camelCase参数转换为算力节点snake_case格式后转发
4. 同步返回算力节点的诊断结果给后端
5. 完整日志记录全链路交互

处理流程：
- 诊断推理通过NodeManager转发到算力节点
- 算力节点调用LLM API并返回诊断结果
- 网关同步返回结果，超时60秒

对应后端接口：POST /badge/v1/algorithm/users/diagnosis-summary
算力节点内部接口：POST /badge/v1/internal/algorithm/inference/diagnosis-summary
"""
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from config import (
    DATE_FORMAT,
    DIAGNOSIS_INTERNAL_PATH,
    STORE_DIAGNOSIS_INTERNAL_PATH,
    DIAGNOSIS_REQUEST_TIMEOUT,
    REQUEST_ID_HEADER,
)
from node_manager import NodeManager
from http_client import HttpClientSingleton
from exception import ErrorCode, ErrorMsg, build_error_response
from utils import TimeFormatter


class DiagnosisHandler:
    """
    AI时段诊断总结请求处理器 - 单例模式

    职责：
    - 接收后端主动调用的诊断请求
    - 校验入参必传字段（对齐v3.1文档6.1节）
    - 生成requestId，通过NodeManager负载均衡选择算力节点
    - 将后端camelCase字段转换为算力节点snake_case格式
    - 转发请求到算力节点并同步等待结果
    - 完整日志记录全链路交互

    设计约束：
    - 同步返回结果，不异步回调
    - 超时时间60秒
    - 必须使用NodeManager实现负载均衡
    - 后端传camelCase，算力节点接收snake_case，网关负责转换
    """

    _instance: Optional["DiagnosisHandler"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._node_manager: Optional[NodeManager] = None
        self._initialized = False

    def initialize(self, node_manager: NodeManager) -> None:
        """
        初始化诊断处理器
        在FastAPI lifespan startup阶段调用

        Args:
            node_manager: 节点管理器实例（已启动健康检查）
        """
        if self._initialized:
            return
        self._node_manager = node_manager
        self._initialized = True
        logger.info(
            f"AI诊断处理器初始化完成 | "
            f"内部路径={DIAGNOSIS_INTERNAL_PATH} | "
            f"超时={DIAGNOSIS_REQUEST_TIMEOUT}s"
        )

    async def handle_diagnosis(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理AI时段诊断总结请求
        完整流程：入参校验 → 选择算力节点 → 转发 → 返回结果

        Args:
            request_body: 后端请求体（camelCase，对齐v3.1文档6.1节）

        Returns:
            标准响应体
        """
        if not self._initialized or self._node_manager is None:
            logger.error("AI诊断处理器未初始化，拒绝请求")
            return build_error_response(
                code=ErrorCode.INTERNAL_ERROR,
                msg="诊断服务未就绪",
            )

        # 入参校验
        validation_error = self._validate_request(request_body)
        if validation_error:
            return validation_error

        request_id = str(uuid.uuid4())
        employee_no = request_body.get("employeeNo", "")
        start_date = request_body.get("startDate", "")
        end_date = request_body.get("endDate", "")

        logger.info(
            f"AI诊断请求开始 | request_id={request_id} | "
            f"员工={employee_no} | 时间范围={start_date} ~ {end_date}"
        )

        # ========== 选择算力节点 ==========
        selected_node = self._node_manager.get_least_connection_node()
        if selected_node is None:
            logger.error(
                f"AI诊断请求失败 | 无可用算力节点 | request_id={request_id}"
            )
            return build_error_response(
                code=ErrorCode.NO_AVAILABLE_NODE,
                msg=ErrorMsg.NO_AVAILABLE_NODE,
                request_id=request_id,
            )

        # ========== 构建算力节点请求体（camelCase → snake_case） ==========
        internal_body = self._build_internal_body(request_body, request_id)
        target_url = f"http://{selected_node}{DIAGNOSIS_INTERNAL_PATH}"

        start_ts = time.time()
        await self._node_manager.increment_connection(selected_node)

        try:
            client = await HttpClientSingleton.get_client()

            forward_headers = {
                REQUEST_ID_HEADER: request_id,
                "Content-Type": "application/json",
            }

            logger.info(
                f"AI诊断转发开始 | request_id={request_id} | "
                f"目标={target_url} | 节点={selected_node}"
            )

            # ========== 同步等待算力节点返回 ==========
            response = await client.post(
                url=target_url,
                json=internal_body,
                headers=forward_headers,
                timeout=DIAGNOSIS_REQUEST_TIMEOUT,
            )

            elapsed_ms = int((time.time() - start_ts) * 1000)

            if 200 <= response.status_code < 300:
                result = response.json()

                # 算力节点返回snake_case，转换为camelCase给后端
                result_data = result.get("data", {})
                if isinstance(result_data, dict):
                    # 转换dimension_type → dimensionType, dimension_code → dimensionCode
                    dimensions = result_data.get("dimensions", [])
                    normalized_dimensions = []
                    for dim in dimensions:
                        if not isinstance(dim, dict):
                            continue
                        if "dimension_type" in dim and "dimensionType" not in dim:
                            dim["dimensionType"] = dim.pop("dimension_type")
                        if "dimension_code" in dim and "dimensionCode" not in dim:
                            dim["dimensionCode"] = dim.pop("dimension_code")
                        normalized_dimensions.append(
                            {
                                "dimensionCode": dim.get("dimensionCode", ""),
                                "dimensionType": dim.get("dimensionType", ""),
                                "summary": dim.get("summary", ""),
                                "suggestion": dim.get("suggestion") or "",
                            }
                        )
                    result_data["dimensions"] = normalized_dimensions

                logger.info(
                    f"AI诊断请求成功 | request_id={request_id} | "
                    f"节点={selected_node} | 耗时={elapsed_ms}ms | "
                    f"员工={employee_no}"
                )

                # 确保响应包含request_id
                if isinstance(result, dict):
                    result["request_id"] = request_id
                return result
            else:
                error_body = response.text[:500]
                logger.error(
                    f"AI诊断请求失败 | request_id={request_id} | "
                    f"节点={selected_node} | 状态码={response.status_code} | "
                    f"响应={error_body} | 耗时={elapsed_ms}ms"
                )
                return build_error_response(
                    code=ErrorCode.NODE_REQUEST_FAILED,
                    msg=ErrorMsg.NODE_REQUEST_FAILED,
                    request_id=request_id,
                )

        except httpx.TimeoutException as e:
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.error(
                f"AI诊断请求超时 | request_id={request_id} | "
                f"节点={selected_node} | 超时类型={type(e).__name__} | "
                f"耗时={elapsed_ms}ms | 超时阈值={DIAGNOSIS_REQUEST_TIMEOUT}s"
            )
            return build_error_response(
                code=ErrorCode.GATEWAY_TIMEOUT,
                msg=f"AI诊断请求超时({DIAGNOSIS_REQUEST_TIMEOUT}s)",
                request_id=request_id,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.exception(
                f"AI诊断请求异常 | request_id={request_id} | "
                f"节点={selected_node} | 异常类型={type(e).__name__} | "
                f"耗时={elapsed_ms}ms"
            )
            return build_error_response(
                code=ErrorCode.NODE_REQUEST_FAILED,
                msg=ErrorMsg.NODE_REQUEST_FAILED,
                request_id=request_id,
            )

        finally:
            await self._node_manager.decrement_connection(selected_node)

    async def handle_store_diagnosis(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理门店AI时段诊断总结请求。
        """
        if not self._initialized or self._node_manager is None:
            logger.error("门店AI诊断处理器未初始化，拒绝请求")
            return build_error_response(
                code=ErrorCode.INTERNAL_ERROR,
                msg="诊断服务未就绪",
            )

        validation_error = self._validate_store_request(request_body)
        if validation_error:
            return validation_error

        request_id = str(uuid.uuid4())
        store_id = request_body.get("storeId", "")
        store_name = request_body.get("storeName", "")
        start_date = request_body.get("startDate", "")
        end_date = request_body.get("endDate", "")
        behaviors = request_body.get("behaviors", [])

        logger.info(
            f"门店AI诊断请求开始 | request_id={request_id} | "
            f"门店={store_id}/{store_name} | 时间范围={start_date} ~ {end_date} | "
            f"行为数={len(behaviors)}"
        )

        selected_node = self._node_manager.get_least_connection_node()
        if selected_node is None:
            logger.error(
                f"门店AI诊断请求失败 | 无可用算力节点 | request_id={request_id}"
            )
            return build_error_response(
                code=ErrorCode.NO_AVAILABLE_NODE,
                msg=ErrorMsg.NO_AVAILABLE_NODE,
                request_id=request_id,
            )

        internal_body = self._build_store_internal_body(request_body, request_id)
        target_url = f"http://{selected_node}{STORE_DIAGNOSIS_INTERNAL_PATH}"

        start_ts = time.time()
        await self._node_manager.increment_connection(selected_node)

        try:
            client = await HttpClientSingleton.get_client()
            forward_headers = {
                REQUEST_ID_HEADER: request_id,
                "Content-Type": "application/json",
            }

            logger.info(
                f"门店AI诊断转发开始 | request_id={request_id} | "
                f"目标={target_url} | 节点={selected_node}"
            )

            response = await client.post(
                url=target_url,
                json=internal_body,
                headers=forward_headers,
                timeout=DIAGNOSIS_REQUEST_TIMEOUT,
            )

            elapsed_ms = int((time.time() - start_ts) * 1000)

            if 200 <= response.status_code < 300:
                result = response.json()
                if isinstance(result, dict):
                    result["request_id"] = request_id

                logger.info(
                    f"门店AI诊断请求成功 | request_id={request_id} | "
                    f"节点={selected_node} | 耗时={elapsed_ms}ms | 门店={store_id}"
                )
                return result

            error_body = response.text[:500]
            logger.error(
                f"门店AI诊断请求失败 | request_id={request_id} | "
                f"节点={selected_node} | 状态码={response.status_code} | "
                f"响应={error_body} | 耗时={elapsed_ms}ms"
            )
            return build_error_response(
                code=ErrorCode.NODE_REQUEST_FAILED,
                msg=ErrorMsg.NODE_REQUEST_FAILED,
                request_id=request_id,
            )

        except httpx.TimeoutException as e:
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.error(
                f"门店AI诊断请求超时 | request_id={request_id} | "
                f"节点={selected_node} | 超时类型={type(e).__name__} | "
                f"耗时={elapsed_ms}ms | 超时阈值={DIAGNOSIS_REQUEST_TIMEOUT}s"
            )
            return build_error_response(
                code=ErrorCode.GATEWAY_TIMEOUT,
                msg=f"门店AI诊断请求超时({DIAGNOSIS_REQUEST_TIMEOUT}s)",
                request_id=request_id,
            )

        except Exception:
            elapsed_ms = int((time.time() - start_ts) * 1000)
            logger.exception(
                f"门店AI诊断请求异常 | request_id={request_id} | "
                f"节点={selected_node} | 耗时={elapsed_ms}ms"
            )
            return build_error_response(
                code=ErrorCode.NODE_REQUEST_FAILED,
                msg=ErrorMsg.NODE_REQUEST_FAILED,
                request_id=request_id,
            )

        finally:
            await self._node_manager.decrement_connection(selected_node)

    # ==================== 参数转换：camelCase → snake_case ====================

    @staticmethod
    def _build_internal_body(
        request_body: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """
        将后端camelCase参数转换为算力节点snake_case格式
        网关的核心职责之一：协议转换

        后端字段 → 算力节点字段映射：
        - employeeNo → employee_no
        - startDate → start_date
        - endDate → end_date
        - score → score
        - dimensionScores → dimension_scores
        - behaviorStats → behavior_stats
        - abnormalBehaviors → abnormal_behaviors
        - （新增）request_id

        Args:
            request_body: 后端发送的请求体（camelCase）
            request_id: 网关生成的请求ID

        Returns:
            算力节点内部请求体（snake_case）
        """
        # 转换dimensionScores中每项的dimensionCode → dimension_code
        dimension_scores = request_body.get("dimensionScores", [])
        converted_scores = []
        for dim in dimension_scores:
            converted_dim = {
                "dimension_code": dim.get("dimensionCode", ""),
                "dimension_name": dim.get("dimensionName", ""),
                "score": dim.get("score", 0),
                "avg_score": dim.get("avgScore", 0),
            }
            converted_scores.append(converted_dim)

        # 转换abnormalBehaviors中每项的字段名
        abnormal_behaviors = request_body.get("abnormalBehaviors", [])
        converted_abnormal = []
        for ab in abnormal_behaviors:
            converted_ab = {
                "event_time": ab.get("eventTime", ""),
                "summary": ab.get("summary", ""),
            }
            converted_abnormal.append(converted_ab)

        # 转换behaviorStats
        behavior_stats = request_body.get("behaviorStats", {})
        converted_stats = {
            "standard_count": behavior_stats.get("standardCount", 0),
            "abnormal_count": behavior_stats.get("abnormalCount", 0),
            "customer_count": behavior_stats.get("customerCount", 0),
        }

        internal_body = {
            "employee_no": request_body.get("employeeNo", ""),
            "start_date": request_body.get("startDate", ""),
            "end_date": request_body.get("endDate", ""),
            "score": request_body.get("score", 0),
            "dimension_scores": converted_scores,
            "behavior_stats": converted_stats,
            "abnormal_behaviors": converted_abnormal,
            "request_id": request_id,
        }
        return internal_body

    @staticmethod
    def _build_store_internal_body(
        request_body: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """将门店诊断请求从camelCase转换为算力节点snake_case格式"""
        converted_behaviors = []
        for behavior in request_body.get("behaviors", []):
            converted_behaviors.append(
                {
                    "behavior_event_id": behavior.get("behaviorEventId", ""),
                    "behavior_type": behavior.get("behaviorType", ""),
                    "event_time": behavior.get("eventTime", ""),
                    "employee_id": behavior.get("employeeId", ""),
                    "employee_name": behavior.get("employeeName", ""),
                    "device_no": behavior.get("deviceNo", ""),
                    "config_item_id": behavior.get("configItemId", ""),
                    "config_item_name": behavior.get("configItemName", ""),
                    "keyword_content": behavior.get("keywordContent", ""),
                    "summary": behavior.get("summary", ""),
                    "review_status": behavior.get("reviewStatus", ""),
                }
            )

        return {
            "store_id": request_body.get("storeId", ""),
            "store_name": request_body.get("storeName", ""),
            "start_date": request_body.get("startDate", ""),
            "end_date": request_body.get("endDate", ""),
            "behaviors": converted_behaviors,
            "request_id": request_id,
        }

    # ==================== 入参校验 ====================

    @staticmethod
    def _validate_request(request_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """校验诊断请求的必传字段（严格对齐v3.1文档6.1节）"""

        # 校验employeeNo
        employee_no = request_body.get("employeeNo")
        if not employee_no or not isinstance(employee_no, str) or not employee_no.strip():
            logger.warning("AI诊断入参校验失败 | employeeNo为空或非字符串")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="employeeNo为必传字段，且必须为非空字符串",
            )

        # 校验startDate
        start_date = request_body.get("startDate")
        if not start_date or not isinstance(start_date, str):
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="startDate为必传字段，格式：yyyy-MM-dd",
            )
        parsed_start = TimeFormatter.parse_date(start_date)
        if not parsed_start:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg=f"startDate格式不合法，要求：yyyy-MM-dd，实际：{start_date}",
            )

        # 校验endDate
        end_date = request_body.get("endDate")
        if not end_date or not isinstance(end_date, str):
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="endDate为必传字段，格式：yyyy-MM-dd",
            )
        parsed_end = TimeFormatter.parse_date(end_date)
        if not parsed_end:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg=f"endDate格式不合法，要求：yyyy-MM-dd，实际：{end_date}",
            )

        if parsed_end < parsed_start:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="endDate不能早于startDate",
            )

        # 校验score
        score = request_body.get("score")
        if score is None:
            return build_error_response(code=ErrorCode.BAD_REQUEST, msg="score为必传字段")
        if not isinstance(score, (int, float)):
            return build_error_response(code=ErrorCode.BAD_REQUEST, msg="score必须为数值类型")

        # 校验dimensionScores
        dimension_scores = request_body.get("dimensionScores")
        if not dimension_scores or not isinstance(dimension_scores, list):
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="dimensionScores为必传字段，且必须为非空数组",
            )
        if len(dimension_scores) == 0:
            return build_error_response(code=ErrorCode.BAD_REQUEST, msg="dimensionScores不能为空数组")
        for i, dim in enumerate(dimension_scores):
            if not isinstance(dim, dict):
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]必须为对象",
                )
            if "dimensionCode" not in dim:
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]缺少dimensionCode字段",
                )
            # dimensionCode为后端配置项ID（如"config-item-001"），仅校验存在且为非空字符串
            if not isinstance(dim["dimensionCode"], str) or not dim["dimensionCode"].strip():
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}].dimensionCode必须为非空字符串",
                )
            if "dimensionName" not in dim:
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]缺少dimensionName字段",
                )
            if not isinstance(dim["dimensionName"], str) or not dim["dimensionName"].strip():
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}].dimensionName必须为非空字符串",
                )
            if "score" not in dim:
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]缺少score字段",
                )
            if "avgScore" not in dim:
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]缺少avgScore字段",
                )
            if not isinstance(dim["score"], (int, float)):
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}].score必须为数值类型",
                )
            if not isinstance(dim["avgScore"], (int, float)):
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}].avgScore必须为数值类型",
                )

        # 校验behaviorStats
        behavior_stats = request_body.get("behaviorStats")
        if not behavior_stats or not isinstance(behavior_stats, dict):
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="behaviorStats为必传字段，且必须为对象",
            )
        for stat_key in ["standardCount", "abnormalCount", "customerCount"]:
            if stat_key not in behavior_stats:
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"behaviorStats缺少{stat_key}字段",
                )

        # 校验abnormalBehaviors
        abnormal_behaviors = request_body.get("abnormalBehaviors")
        if abnormal_behaviors is None:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="abnormalBehaviors为必传字段（可为空数组）",
            )
        if not isinstance(abnormal_behaviors, list):
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="abnormalBehaviors必须为数组",
            )
        for i, behavior in enumerate(abnormal_behaviors):
            if not isinstance(behavior, dict):
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"abnormalBehaviors[{i}]必须为对象",
                )
            for req_key in ["eventTime", "summary"]:
                if req_key not in behavior:
                    return build_error_response(
                        code=ErrorCode.BAD_REQUEST,
                        msg=f"abnormalBehaviors[{i}]缺少{req_key}字段",
                    )

        return None  # 校验通过

    @staticmethod
    def _validate_store_request(request_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """校验门店诊断请求必传字段"""
        for key in ["storeId", "storeName", "startDate", "endDate"]:
            value = request_body.get(key)
            if not value or not isinstance(value, str) or not value.strip():
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"{key}为必传字段，且必须为非空字符串",
                )

        start_date = request_body.get("startDate", "")
        parsed_start = TimeFormatter.parse_date(start_date)
        if not parsed_start:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg=f"startDate格式不合法，要求：yyyy-MM-dd，实际：{start_date}",
            )

        end_date = request_body.get("endDate", "")
        parsed_end = TimeFormatter.parse_date(end_date)
        if not parsed_end:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg=f"endDate格式不合法，要求：yyyy-MM-dd，实际：{end_date}",
            )

        if parsed_end < parsed_start:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="endDate不能早于startDate",
            )

        behaviors = request_body.get("behaviors")
        if behaviors is None:
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="behaviors为必传字段（可为空数组）",
            )
        if not isinstance(behaviors, list):
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="behaviors必须为数组",
            )

        required_behavior_keys = [
            "behaviorEventId",
            "behaviorType",
            "eventTime",
            "employeeId",
            "employeeName",
            "deviceNo",
            "configItemId",
            "configItemName",
            "keywordContent",
            "summary",
            "reviewStatus",
        ]
        valid_behavior_types = {"STANDARD", "ABNORMAL", "CUSTOMER"}

        for index, behavior in enumerate(behaviors):
            if not isinstance(behavior, dict):
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"behaviors[{index}]必须为对象",
                )
            for key in required_behavior_keys:
                value = behavior.get(key)
                if not value or not isinstance(value, str) or not value.strip():
                    return build_error_response(
                        code=ErrorCode.BAD_REQUEST,
                        msg=f"behaviors[{index}].{key}为必传字段，且必须为非空字符串",
                    )
            behavior_type = behavior.get("behaviorType")
            if behavior_type not in valid_behavior_types:
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=(
                        f"behaviors[{index}].behaviorType无效，"
                        "仅支持STANDARD/ABNORMAL/CUSTOMER"
                    ),
                )

        return None

    def is_initialized(self) -> bool:
        """诊断处理器是否已初始化"""
        return self._initialized
