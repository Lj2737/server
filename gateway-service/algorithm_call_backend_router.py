"""
算法调用后端接口。

仅保留交付物清单中算法侧直接调用后端的接口：
- 5.2 语音行为识别回调
- 5.3 AI 对话完成回调
- 6.4 算法查询设备 FastGPT 知识库 ID
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from config import (
    BEHAVIOR_CALLBACK_PATH,
    DIALOG_COMPLETION_CALLBACK_PATH,
    KNOWLEDGE_BASE_QUERY_PATH,
)
from utils import BackendClient


class DialogCompletionRequest(BaseModel):
    """AI 对话完成回调请求。"""
    deviceNo: str = Field(..., min_length=1, description="设备编号", examples=["BADGE0001"])
    dialogTime: str = Field(
        ...,
        min_length=1,
        description="对话完成时间，格式 yyyy-MM-dd HH:mm:ss",
        examples=["2026-05-08 10:35:00"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "deviceNo": "BADGE0001",
                    "dialogTime": "2026-05-08 10:35:00",
                }
            ]
        }
    }


class KnowledgeBaseQueryRequest(BaseModel):
    """设备知识库 ID 查询请求。"""
    deviceNo: str = Field(..., min_length=1, description="设备编号", examples=["BADGE0001"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "deviceNo": "BADGE0001",
                }
            ]
        }
    }


class StandardResponse(BaseModel):
    """统一响应。"""
    code: int = Field(200, description="业务状态码")
    msg: str = Field("ok", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")


class KnowledgeBaseResponse(BaseModel):
    """知识库 ID 查询响应。"""
    knowledgeBaseId: Optional[str] = Field(None, description="FastGPT 知识库 ID")


algorithm_call_backend_router = APIRouter(tags=["算法调用后端接口"])


@algorithm_call_backend_router.post(
    BEHAVIOR_CALLBACK_PATH,
    response_model=StandardResponse,
    summary="语音行为识别回调",
)
async def voice_behavior_callback(
    metadata: str = Form(
        ...,
        description=(
            "行为识别元数据 JSON 字符串，包含 eventTime、deviceNo、behaviorType、"
            "summary、configItemId、keywordContent"
        ),
        examples=[
            '{"eventTime":"2026-05-08 10:31:00","deviceNo":"BADGE0001",'
            '"behaviorType":"ABNORMAL","summary":"员工触发服务禁语",'
            '"configItemId":"forbidden-service-attitude","keywordContent":"你自己看"}'
        ],
    ),
    file: Optional[UploadFile] = File(
        None,
        description="异常行为语音片段；behaviorType=ABNORMAL 时必传",
    ),
):
    """接收算法语音行为识别结果，并转发后端。"""
    files = None
    if file is not None:
        file_bytes = await file.read()
        files = {
            "file": (
                file.filename or "voice-behavior.wav",
                file_bytes,
                file.content_type or "application/octet-stream",
            )
        }

    result = await BackendClient().post_multipart(
        path=BEHAVIOR_CALLBACK_PATH,
        files=files,
        data={"metadata": metadata},
    )
    return result


@algorithm_call_backend_router.post(
    DIALOG_COMPLETION_CALLBACK_PATH,
    response_model=StandardResponse,
    summary="AI 对话完成回调",
)
async def dialog_completion_callback(request: DialogCompletionRequest):
    """接收算法 AI 对话完成通知，并转发后端。"""
    result = await BackendClient().post(
        path=DIALOG_COMPLETION_CALLBACK_PATH,
        json_body=request.model_dump(),
    )
    return result


@algorithm_call_backend_router.post(
    KNOWLEDGE_BASE_QUERY_PATH,
    response_model=KnowledgeBaseResponse,
    summary="查询设备知识库 ID",
)
async def query_knowledge_base(request: KnowledgeBaseQueryRequest):
    """按设备编号查询 FastGPT 知识库 ID。"""
    knowledge_base_id = await BackendClient().get_knowledge_base_id(request.deviceNo)
    return {"knowledgeBaseId": knowledge_base_id}
