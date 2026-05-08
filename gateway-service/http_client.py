"""
智能胸牌服务管理系统 - httpx异步客户端单例封装
复用TCP连接池，减少建连开销，提升转发性能
适配树莓派资源，控制连接池大小
"""
import httpx

from config import REQUEST_TIMEOUT, HTTP_CLIENT_MAX_CONNECTIONS, HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS


class HttpClientSingleton:
    """
    httpx.AsyncClient单例封装
    - 复用TCP连接池，减少每次请求的建连开销
    - 统一超时配置，与config.py保持一致
    - 懒加载模式，首次使用时创建实例
    - 提供显式关闭方法，配合FastAPI生命周期管理
    """
    _instance: httpx.AsyncClient | None = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        """
        获取httpx异步客户端单例
        - 如果实例不存在或已关闭，则创建新实例
        - 存在且有效则直接返回，实现连接池复用

        Returns:
            httpx.AsyncClient: 异步HTTP客户端实例
        """
        if cls._instance is None or cls._instance.is_closed:
            cls._instance = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,              # 建立TCP连接超时5秒
                    read=REQUEST_TIMEOUT,     # 读取响应超时，使用配置值
                    write=10.0,               # 写入请求体超时10秒
                    pool=5.0,                 # 从连接池获取连接超时5秒
                ),
                limits=httpx.Limits(
                    max_connections=HTTP_CLIENT_MAX_CONNECTIONS,             # 总连接池大小
                    max_keepalive_connections=HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS,  # 保持活跃连接数
                ),
                follow_redirects=False,  # 禁止重定向，避免请求被意外转发
            )
        return cls._instance

    @classmethod
    async def close_client(cls) -> None:
        """
        关闭httpx客户端，释放连接池资源
        在FastAPI shutdown生命周期中调用，确保资源正确释放
        """
        if cls._instance is not None and not cls._instance.is_closed:
            await cls._instance.aclose()
            cls._instance = None
