"""
智能胸牌服务管理系统 - AI时段诊断总结请求处理器
核心功能：
1. 接收后端主动调用，校验入参
2. 生成requestId，通过NodeManager选择可用算力节点
3. 将后端camelCase参数转换为算力节点snake_case格式后转发
4. 同步返回算力节点的诊断结果给后端（超时30秒）
5. 完整日志记录全链路交互

对应后端接口：POST /algorithm/badge/users/diagnosis-summary
算力节点内部接口：POST /api/v1/internal/inference/diagnosis-summary

处理流程：
    步骤1：入参校验，必传字段：employeeNo、startDate、endDate、score、dimensionScores、behaviorStats、abnormalBehaviors
    步骤2：生成requestId，调用NodeManager选择可用的算力节点
    步骤3：将后端camelCase字段转换为算力节点snake_case格式，转发到算力节点内部接口
    步骤4：收到算力节点返回的结果后，原封不动返回给后端
    步骤5：同步返回结果，超时时间设置为30秒
    步骤6：日志完整记录：员工编号、时间范围、请求参数、返回结果

请求体格式（后端→网关）：
    {
        "employeeNo": "EMP001",
        "startDate": "2026-05-02",
        "endDate": "2026-05-08",
        "score": 86,
        "dimensionScores": [
            {"dimensionCode": "SERVICE_RESPONSE", "score": 82}
        ],
        "behaviorStats": {
            "standardCount": 12,
            "abnormalCount": 2,
            "customerCount": 1
        },
        "abnormalBehaviors": [
            {
                "behaviorEventId": "AI_BEHAVIOR_20260508103100001",
                "eventTime": "2026-05-08 10:31:00",
                "summary": "员工未按SOP回应顾客问题"
            }
        ]
    }

算力节点内部请求体格式（网关→算力节点，snake_case）：
    {
        "employee_no": "EMP001",
        "start_date": "2026-05-02",
        "end_date": "2026-05-08",
        "score": 86,
        "dimension_scores": [...],
        "behavior_stats": {...},
        "abnormal_behaviors": [...],
        "request_id": "req-diag-xxxx"
    }

使用示例：
    from diagnosis_handler import DiagnosisHandler
    from node_manager import NodeManager

    # 获取单例（由main.py在启动时初始化）
    handler = DiagnosisHandler()

    # 处理诊断请求
    result = await handler.handle_diagnosis(request_body={
        "employeeNo": "EMP001",
        "startDate": "2026-05-02",
        "endDate": "2026-05-08",
        "score": 86,
        "dimensionScores": [{"dimensionCode": "SERVICE_RESPONSE", "score": 82}],
        "behaviorStats": {"standardCount": 12, "abnormalCount": 2, "customerCount": 1},
        "abnormalBehaviors": [],
    })
"""
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from config import (
    DIAGNOSIS_INTERNAL_PATH,
    DIAGNOSIS_REQUEST_TIMEOUT,
    REQUEST_ID_HEADER,
    DATE_FORMAT,
)
from node_manager import NodeManager
from http_client import HttpClientSingleton
from exception import GatewayException, ErrorCode, ErrorMsg, build_error_response
from utils import TimeFormatter


class DiagnosisHandler:
    """
    AI时段诊断总结请求处理器 - 单例模式

    职责：
    - 接收后端主动调用的诊断请求
    - 校验入参必传字段
    - 将后端camelCase字段转换为算力节点snake_case格式
    - 通过NodeManager负载均衡选择算力节点
    - 转发请求到算力节点并同步等待结果
    - 完整日志记录全链路交互

    设计约束：
    - 同步返回结果，不异步回调（区别于行为识别的fire-and-forget）
    - 超时时间30秒（LLM推理耗时较长）
    - 必须使用NodeManager实现负载均衡
    - 必须使用TimeFormatter统一时间格式
    - 后端传camelCase，算力节点接收snake_case，网关负责转换
    """

    _instance: Optional["DiagnosisHandler"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个DiagnosisHandler实例"""
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

    # ==================== 核心处理方法 ====================

    async def handle_diagnosis(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理AI时段诊断总结请求
        完整实现步骤1~6，由backend_router调用

        Args:
            request_body: 后端发送的请求体（camelCase）
                {
                    "employeeNo": "EMP001",
                    "startDate": "2026-05-02",
                    "endDate": "2026-05-08",
                    "score": 86,
                    "dimensionScores": [...],
                    "behaviorStats": {...},
                    "abnormalBehaviors": [...]
                }

        Returns:
            标准响应体：
            成功：{"code": 200, "msg": "ok", "data": {...诊断结果...}, "request_id": "..."}
            失败：{"code": 错误码, "msg": "错误信息", "data": null, "request_id": "..."}
        """
        if not self._initialized or self._node_manager is None:
            logger.error("AI诊断处理器未初始化，拒绝请求")
            return build_error_response(
                code=ErrorCode.INTERNAL_ERROR,
                msg="诊断服务未就绪",
            )

        # ========== 步骤1：入参校验（优先校验，给调用方明确的错误信息） ==========
        validation_error = self._validate_request(request_body)
        if validation_error:
            return validation_error

        # ========== 步骤2：生成requestId + 选择算力节点 ==========
        request_id = str(uuid.uuid4())
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

        # 提取关键参数用于日志
        employee_no = request_body.get("employeeNo", "")
        start_date = request_body.get("startDate", "")
        end_date = request_body.get("endDate", "")

        logger.info(
            f"AI诊断请求开始 | request_id={request_id} | "
            f"节点={selected_node} | 员工={employee_no} | "
            f"时间范围={start_date} ~ {end_date}"
        )

        # ========== 步骤3：将后端camelCase参数转换为算力节点snake_case格式并转发 ==========
        # 构建算力节点内部请求体：snake_case字段 + requestId
        internal_body = self._build_internal_body(request_body, request_id)

        # 构建算力节点完整URL
        target_url = f"http://{selected_node}{DIAGNOSIS_INTERNAL_PATH}"

        start_ts = time.time()
        await self._node_manager.increment_connection(selected_node)

        try:
            client = await HttpClientSingleton.get_client()

            # 构建转发请求头
            forward_headers = {
                REQUEST_ID_HEADER: request_id,
                "Content-Type": "application/json",
            }

            logger.info(
                f"AI诊断转发开始 | request_id={request_id} | "
                f"目标={target_url} | 节点={selected_node}"
            )

            # ========== 步骤4+5：同步等待算力节点返回，超时30秒 ==========
            response = await client.post(
                url=target_url,
                json=internal_body,
                headers=forward_headers,
                timeout=DIAGNOSIS_REQUEST_TIMEOUT,
            )

            elapsed_ms = int((time.time() - start_ts) * 1000)

            # 处理算力节点响应
            if 200 <= response.status_code < 300:
                result = response.json()
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
                # 算力节点返回非2xx
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
                msg=f"诊断请求超时({DIAGNOSIS_REQUEST_TIMEOUT}s)",
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
            # 无论成功或失败，必须减少节点活跃连接数
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
        internal_body = {
            "employee_no": request_body.get("employeeNo", ""),
            "start_date": request_body.get("startDate", ""),
            "end_date": request_body.get("endDate", ""),
            "score": request_body.get("score", 0),
            "dimension_scores": request_body.get("dimensionScores", []),
            "behavior_stats": request_body.get("behaviorStats", {}),
            "abnormal_behaviors": request_body.get("abnormalBehaviors", []),
            "request_id": request_id,
        }
        return internal_body

    # ==================== 入参校验 ====================

    @staticmethod
    def _validate_request(request_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        校验诊断请求的必传字段

        必传字段：
        - employeeNo: 员工编号，非空字符串
        - startDate: 诊断开始日期，格式yyyy-MM-dd
        - endDate: 诊断结束日期，格式yyyy-MM-dd
        - score: 时间段评分，数值
        - dimensionScores: 维度评分列表，至少1个
        - behaviorStats: 行为统计，必须包含standardCount/abnormalCount/customerCount
        - abnormalBehaviors: 异常行为列表，可为空数组

        Args:
            request_body: 请求体

        Returns:
            None: 校验通过
            Dict: 校验失败的错误响应
        """
        # 校验employeeNo（单数，单个员工编号）
        employee_no = request_body.get("employeeNo")
        if not employee_no or not isinstance(employee_no, str) or not employee_no.strip():
            logger.warning("AI诊断入参校验失败 | employeeNo为空或非字符串")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="employeeNo为必传字段，且必须为非空字符串",
            )

        # 校验startDate（日期格式yyyy-MM-dd）
        start_date = request_body.get("startDate")
        if not start_date or not isinstance(start_date, str):
            logger.warning("AI诊断入参校验失败 | startDate为空或非字符串")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="startDate为必传字段，格式：yyyy-MM-dd",
            )
        parsed_start = TimeFormatter.parse_date(start_date)
        if not parsed_start:
            logger.warning(f"AI诊断入参校验失败 | startDate格式不合法 | 值={start_date}")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg=f"startDate格式不合法，要求：yyyy-MM-dd，实际：{start_date}",
            )

        # 校验endDate（日期格式yyyy-MM-dd）
        end_date = request_body.get("endDate")
        if not end_date or not isinstance(end_date, str):
            logger.warning("AI诊断入参校验失败 | endDate为空或非字符串")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="endDate为必传字段，格式：yyyy-MM-dd",
            )
        parsed_end = TimeFormatter.parse_date(end_date)
        if not parsed_end:
            logger.warning(f"AI诊断入参校验失败 | endDate格式不合法 | 值={end_date}")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg=f"endDate格式不合法，要求：yyyy-MM-dd，实际：{end_date}",
            )

        # 校验日期范围：endDate必须晚于或等于startDate
        if parsed_end < parsed_start:
            logger.warning(
                f"AI诊断入参校验失败 | endDate早于startDate | "
                f"start={start_date} | end={end_date}"
            )
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="endDate不能早于startDate",
            )

        # 校验score（时间段评分）
        score = request_body.get("score")
        if score is None:
            logger.warning("AI诊断入参校验失败 | score为空")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="score为必传字段",
            )
        if not isinstance(score, (int, float)):
            logger.warning(f"AI诊断入参校验失败 | score非数值 | 值={score}")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="score必须为数值类型",
            )

        # 校验dimensionScores（维度评分列表）
        dimension_scores = request_body.get("dimensionScores")
        if not dimension_scores or not isinstance(dimension_scores, list):
            logger.warning("AI诊断入参校验失败 | dimensionScores为空或非数组")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="dimensionScores为必传字段，且必须为非空数组",
            )
        if len(dimension_scores) == 0:
            logger.warning("AI诊断入参校验失败 | dimensionScores数组为空")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="dimensionScores不能为空数组",
            )
        # 校验每个维度评分项包含dimensionCode和score
        for i, dim in enumerate(dimension_scores):
            if not isinstance(dim, dict):
                logger.warning(f"AI诊断入参校验失败 | dimensionScores[{i}]非对象")
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]必须为对象，包含dimensionCode和score",
                )
            if "dimensionCode" not in dim:
                logger.warning(f"AI诊断入参校验失败 | dimensionScores[{i}]缺少dimensionCode")
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]缺少dimensionCode字段",
                )
            if "score" not in dim:
                logger.warning(f"AI诊断入参校验失败 | dimensionScores[{i}]缺少score")
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"dimensionScores[{i}]缺少score字段",
                )

        # 校验behaviorStats（行为统计）
        behavior_stats = request_body.get("behaviorStats")
        if not behavior_stats or not isinstance(behavior_stats, dict):
            logger.warning("AI诊断入参校验失败 | behaviorStats为空或非对象")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="behaviorStats为必传字段，且必须为对象",
            )
        # 校验behaviorStats包含必要的统计字段
        for stat_key in ["standardCount", "abnormalCount", "customerCount"]:
            if stat_key not in behavior_stats:
                logger.warning(f"AI诊断入参校验失败 | behaviorStats缺少{stat_key}")
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"behaviorStats缺少{stat_key}字段",
                )

        # 校验abnormalBehaviors（异常行为列表，可为空数组）
        abnormal_behaviors = request_body.get("abnormalBehaviors")
        if abnormal_behaviors is None:
            logger.warning("AI诊断入参校验失败 | abnormalBehaviors为空")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="abnormalBehaviors为必传字段（可为空数组）",
            )
        if not isinstance(abnormal_behaviors, list):
            logger.warning("AI诊断入参校验失败 | abnormalBehaviors非数组")
            return build_error_response(
                code=ErrorCode.BAD_REQUEST,
                msg="abnormalBehaviors必须为数组",
            )
        # 校验每条异常行为包含必传字段
        for i, behavior in enumerate(abnormal_behaviors):
            if not isinstance(behavior, dict):
                logger.warning(f"AI诊断入参校验失败 | abnormalBehaviors[{i}]非对象")
                return build_error_response(
                    code=ErrorCode.BAD_REQUEST,
                    msg=f"abnormalBehaviors[{i}]必须为对象",
                )
            for req_key in ["behaviorEventId", "eventTime", "summary"]:
                if req_key not in behavior:
                    logger.warning(
                        f"AI诊断入参校验失败 | abnormalBehaviors[{i}]缺少{req_key}"
                    )
                    return build_error_response(
                        code=ErrorCode.BAD_REQUEST,
                        msg=f"abnormalBehaviors[{i}]缺少{req_key}字段",
                    )

        return None  # 校验通过

    # ==================== 状态查询 ====================

    def is_initialized(self) -> bool:
        """诊断处理器是否已初始化"""
        return self._initialized
