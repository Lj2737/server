"""
智能胸牌服务管理系统 - 全局异常处理与错误码定义
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
    """错误码常量，对齐HTTP标准状态码，禁止自定义乱码"""
    # 400 客户端错误
    BAD_REQUEST = 400                   # 请求参数错误
    NOT_FOUND = 404                     # 路由不存在
    VALIDATION_ERROR = 422              # 参数校验失败

    # 服务端错误
    INTERNAL_ERROR = 500                # 内部服务错误

    # 网关特有错误
    NO_AVAILABLE_NODE = 503             # 无可用算力节点（Service Unavailable）
    GATEWAY_TIMEOUT = 504               # 请求转发超时（Gateway Timeout）
    NODE_REQUEST_FAILED = 502           # 算力节点请求失败（Bad Gateway）


# ==================== 错误消息定义（禁止暴露内部实现细节） ====================

class ErrorMsg:
    """错误提示信息，面向调用方，禁止暴露敏感信息和异常栈"""
    BAD_REQUEST = "请求参数错误"
    NOT_FOUND = "请求的路由不存在"
    VALIDATION_ERROR = "请求参数校验失败"
    INTERNAL_ERROR = "服务内部错误"
    NO_AVAILABLE_NODE = "暂无可用算力节点，请稍后重试"
    GATEWAY_TIMEOUT = "请求处理超时，请稍后重试"
    NODE_REQUEST_FAILED = "算力节点请求失败"


# ==================== 统一错误响应构建 ====================

def build_error_response(
    code: int,
    msg: str,
    request_id: str | None = None,
) -> dict:
    """
    构建统一错误响应体
    格式严格为：{"code": 错误码, "msg": "错误提示", "data": null, "request_id": "请求ID"}

    Args:
        code: 错误码，对齐HTTP标准
        msg: 错误提示信息，禁止暴露敏感信息
        request_id: 请求唯一ID，无则自动生成UUID

    Returns:
        统一格式的错误响应字典
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

class GatewayException(Exception):
    """
    网关自定义业务异常
    用于在业务逻辑中主动抛出（如无可用节点、超时等），
    由全局异常处理器统一捕获并返回标准格式响应
    """

    def __init__(self, code: int, msg: str, request_id: str | None = None):
        """
        Args:
            code: 错误码
            msg: 错误提示信息
            request_id: 请求唯一ID
        """
        self.code = code
        self.msg = msg
        self.request_id = request_id or str(uuid.uuid4())
        super().__init__(msg)


# ==================== 全局异常处理器 ====================

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    全局异常处理器 - 捕获所有未处理异常
    - 记录完整异常日志（含异常栈，仅供内部排查）
    - 对外返回统一格式，绝不暴露敏感信息和异常栈
    """
    request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))

    # 记录完整异常日志（含异常栈），供内部排查
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
    """
    参数校验异常处理器 - 捕获FastAPI参数校验失败
    - 记录校验失败详情日志（仅供内部排查）
    - 对外返回简洁的校验失败提示
    """
    request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))

    # 记录校验失败详情（内部日志，不对外暴露）
    logger.warning(
        f"参数校验失败 | 路径={request.url.path} | "
        f"request_id={request_id} | 详情={exc.errors()}"
    )

    # 对外返回简洁提示
    return JSONResponse(
        status_code=422,
        content=build_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            msg=ErrorMsg.VALIDATION_ERROR,
            request_id=request_id,
        ),
    )


async def gateway_exception_handler(
    request: Request, exc: GatewayException
) -> JSONResponse:
    """
    网关业务异常处理器 - 捕获GatewayException
    - 记录业务异常日志
    - 对外返回统一格式的错误响应
    """
    logger.warning(
        f"网关业务异常 | 路径={request.url.path} | "
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
