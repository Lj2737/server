"""
算法转发设备到后端相关接口聚合路由。

本文件只注册“算法 -> 后端”的设备事件转发入口：
- POST /badge/v1/internal/ai/device-events
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict
from datetime import datetime

from utils import BackendClient


class AlgorithmDeviceEventRequest(BaseModel):
    """
    算法转发给后端的设备事件模型。

    与硬件上报给算法的 /badge/v1/internal/hardware/device-events 区分：
    - 硬件心跳 payload 使用 signalLevel
    - 算法转发后端 payload 使用 signalPercent
    """
    deviceNo: str = Field(..., min_length=1, description="设备编号")
    eventType: str = Field(..., description="事件类型：HEARTBEAT/ALARM")
    reportTime: str = Field(..., description="上报时间，格式yyyy-MM-dd HH:mm:ss")
    payload: Dict[str, Any] = Field(..., description="事件数据")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "deviceNo": "BADGE0001",
                    "eventType": "HEARTBEAT",
                    "reportTime": "2026-05-08 10:30:00",
                    "payload": {
                        "batteryLevel": 86,
                        "signalPercent": 75,
                    },
                },
                {
                    "deviceNo": "BADGE0001",
                    "eventType": "ALARM",
                    "reportTime": "2026-05-08 10:31:00",
                    "payload": {
                        "alarmCode": "MIC_ERROR",
                        "alarmStatus": "ACTIVE",
                    },
                },
            ]
        }
    }

    @field_validator("eventType")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if v not in {"HEARTBEAT", "ALARM"}:
            raise ValueError("eventType必须为HEARTBEAT或ALARM")
        return v

    @field_validator("reportTime")
    @classmethod
    def validate_report_time(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("reportTime格式必须为yyyy-MM-dd HH:mm:ss")
        return v

    @model_validator(mode="after")
    def validate_payload(self) -> "AlgorithmDeviceEventRequest":
        if self.eventType == "HEARTBEAT":
            if "batteryLevel" not in self.payload or (
                "signalPercent" not in self.payload and "signalLevel" not in self.payload
            ):
                raise ValueError("HEARTBEAT payload必须包含batteryLevel和signalPercent或signalLevel")
        if self.eventType == "ALARM":
            if "alarmCode" not in self.payload or "alarmStatus" not in self.payload:
                raise ValueError("ALARM payload必须包含alarmCode和alarmStatus")
        return self


algorithm_forward_device_to_backend_router = APIRouter(
    tags=["算法转发设备到后端相关接口"],
)


@algorithm_forward_device_to_backend_router.post(
    "/badge/v1/internal/ai/device-events",
    summary="转发设备状态",
)
async def forward_device_event_to_backend(request: AlgorithmDeviceEventRequest):
    """
    接收算法转发的设备心跳或告警，并转发后端。
    """
    success = await BackendClient().forward_device_event(request.model_dump())
    return {
        "code": 200,
        "msg": "ok",
        "data": {"success": success},
    }
