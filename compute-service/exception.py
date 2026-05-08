"""
智能胸牌服务管理系统 - 算力节点全局异常处理与错误码定义
统一错误响应格式，禁止对外暴露敏感信息和异常栈
错误码严格对齐HTTP标准，禁止自定义乱码
"""
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from config import REQUEST_ID_HEADER


# ==================== 错误码定义（严格对齐HTTP标准） ====================

class ErrorCode:
    """错误码常量，对齐HTTP标准状态码"""
    BAD_REQUEST = 400                   # 请求参数错误
    FORBIDDEN = 403                     # 访问被拒绝（IP白名单）
    NOT_FOUND = 404                     # 路由不存在
    TOO_MANY_REQUESTS = 429             # 并发超限
    VALIDATION_ERROR = 422              # 参数校验失败
    INTERNAL_ERROR = 500                # 内部服务错误
    MODEL_LOAD_FAILED = 503             # 模型加载失败
    ASR_INFERENCE_FAILED = 503          # ASR推理失败
    LLM_INFERENCE_FAILED = 503          # LLM推理失败
    LLM_OUTPUT_INVALID = 503            # LLM输出格式无效


# ==================== 错误消息定义（禁止暴露内部实现细节） ====================

class ErrorMsg:
    """错误提示信息，面向调用方"""
    BAD_REQUEST = "请求参数错误"
    FORBIDDEN = "访问被拒绝，IP不在白名单中"
    NOT_FOUND = "请求的路由不存在"
    TOO_MANY_REQUESTS = "当前并发请求过多，请稍后重试"
    VALIDATION_ERROR = "请求参数校验失败"
    INTERNAL_ERROR = "服务内部错误"
    MODEL_LOAD_FAILED = "模型加载失败，服务不可用"
    ASR_INFERENCE_FAILED = "语音识别推理失败"
    ASR_OUTPUT_EMPTY = "语音识别输出为空"
    LLM_INFERENCE_FAILED = "大语言模型推理失败"
    LLM_OUTPUT_INVALID = "大语言模型输出格式异常"
    AUDIO_FORMAT_INVALID = "音频格式不符合要求"
    CONFIG_TYPE_INVALID = "配置类型无效"


# ==================== 统一错误响应构建 ====================

def build_error_response(
    code: int,
    msg: str,
    request_id: str | None = None,
) -> dict:
    """
    构建统一错误响应体
    格式：{"code": 错误码, "msg": "错误提示", "data": null, "request_id": "请求ID"}
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    return {
        "code": code,
        "msg": msg,
        "data": None,
        "request_id": request_id,
    }


# ==================== 自定义业务异常 ====================

class ComputeException(Exception):
    """算力节点自定义业务异常"""

    def __init__(self, code: int, msg: str, request_id: str | None = None):
        self.code = code
        self.msg = msg
        self.request_id = request_id or str(uuid.uuid4())
        super().__init__(msg)


# ==================== 全局异常处理器 ====================

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理器 - 捕获所有未处理异常"""
    request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))

    # 记录完整异常日志（含异常栈，仅供内部排查）
    logger.exception(
        f"未处理异常 | 路径={request.url.path} | "
        f"request_id={request_id} | 异常类型={type(exc).__name__}"
    )

    # 对外返回统一格式，不暴露异常详情
    return JSONResponse(
        status_code=500,
        content=build_error_response(
            code=ErrorCode.INTERNAL_ERROR,
            msg=ErrorMsg.INTERNAL_ERROR,
            request_id=request_id,
        ),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """参数校验异常处理器"""
    request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))

    logger.warning(
        f"参数校验失败 | 路径={request.url.path} | "
        f"request_id={request_id} | 详情={exc.errors()}"
    )

    return JSONResponse(
        status_code=422,
        content=build_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            msg=ErrorMsg.VALIDATION_ERROR,
            request_id=request_id,
        ),
    )


async def compute_exception_handler(
    request: Request, exc: ComputeException
) -> JSONResponse:
    """算力节点业务异常处理器"""
    logger.warning(
        f"业务异常 | 路径={request.url.path} | "
        f"request_id={exc.request_id} | 错误码={exc.code} | 错误信息={exc.msg}"
    )

    return JSONResponse(
        status_code=exc.code,
        content=build_error_response(
            code=exc.code,
            msg=exc.msg,
            request_id=exc.request_id,
        ),
    )
