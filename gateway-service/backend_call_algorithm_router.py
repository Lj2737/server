"""
后端调用算法接口聚合路由。

本文件只负责把“后端 -> 算法”的对外网关入口集中到一个独立路由下：
- POST /badge/v1/algorithm/users/diagnosis-summary
- POST /badge/v1/algorithm/config/sync
- POST /badge/v1/algorithm/duty-broadcasts/tts

具体业务实现仍分别保留在 backend_router.py 和 duty_broadcast_router.py。
"""
from fastapi import APIRouter

from backend_router import backend_router, diagnosis_handler, config_sync_handler
from duty_broadcast_router import duty_broadcast_router


backend_call_algorithm_router = APIRouter(tags=["后端调用算法接口"])
backend_call_algorithm_router.include_router(backend_router)
backend_call_algorithm_router.include_router(duty_broadcast_router)

