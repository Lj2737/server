"""
智能胸牌服务管理系统 - IP白名单中间件
仅允许主网关节点IP访问算力节点，禁止其他IP请求
确保算力节点不直接对接后端，仅与主网关通信
"""
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from config import GATEWAY_IP_WHITELIST


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP白名单中间件
    - 每个请求到达前校验客户端IP
    - 仅允许主网关节点IP访问，其他IP返回403
    - /health路径也受白名单限制（主网关健康检查来自白名单IP）
    """

    async def dispatch(self, request: Request, call_next):
        """
        请求拦截与IP校验

        Args:
            request: 当前请求对象
            call_next: 下一个中间件/路由处理函数

        Returns:
            正常请求继续处理，非法IP返回403
        """
        # 获取客户端真实IP
        # 优先从X-Forwarded-For获取（经过代理时），否则取直接连接IP
        client_ip = request.headers.get(
            "x-forwarded-for", ""
        ).split(",")[0].strip()

        if not client_ip:
            # 直接连接，取client.host
            client_ip = request.client.host if request.client else "unknown"

        # IP白名单校验
        if client_ip not in GATEWAY_IP_WHITELIST:
            logger.warning(
                f"IP白名单拦截 | 客户端IP={client_ip} | "
                f"路径={request.url.path} | 白名单={GATEWAY_IP_WHITELIST}"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "code": 403,
                    "msg": "访问被拒绝，IP不在白名单中",
                    "data": None,
                    "request_id": "",
                },
            )

        # IP在白名单中，继续处理请求
        response = await call_next(request)
        return response
