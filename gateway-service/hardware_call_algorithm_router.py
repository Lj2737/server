"""
硬件调用算法接口。

硬件只接入网关；网关校验后再转发到内部算力节点或后端。
"""
from fastapi import APIRouter

from config import HARDWARE_API_PREFIX
from hardware_router import hardware_router
from raw_audio_router import raw_audio_router


hardware_call_algorithm_router = APIRouter(tags=["硬件调用算法接口"])
hardware_call_algorithm_router.include_router(hardware_router)
hardware_call_algorithm_router.include_router(
    raw_audio_router,
    prefix=HARDWARE_API_PREFIX,
)

