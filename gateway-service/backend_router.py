"""
智能胸牌服务管理系统 - 后端调用算法的接口注册
核心功能：
1. 功能3：响应后端AI时段诊断总结调用
   - POST /badge/v1/algorithm/users/diagnosis-summary
2. 功能4：接收后端词库配置同步
   - POST /badge/v1/algorithm/config/sync

功能5值班播报已迁移到 duty_broadcast_router.py（TTS API合成 + WebSocket推送）

这些接口是后端主动调用算法的入口（区别于功能1/2的算法主动回调后端）
路由前缀统一为 /badge/v1/algorithm/，对齐文档后端→算法的接口规范

接口注册说明：
- backend_router 由 main.py 在启动时通过 app.include_router() 注册
- DiagnosisHandler、ConfigSyncHandler 由 main.py 在 lifespan 中初始化并注入 NodeManager
- 接口使用 Pydantic 模型做请求体校验，FastAPI 自动生成 Swagger 文档
- 请求体字段名使用 camelCase，对齐v3文档后端→算法的接口规范
"""
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from diagnosis_handler import DiagnosisHandler
from config_sync_handler import ConfigSyncHandler


# ==================== Pydantic 请求模型 ====================

class DimensionScoreItem(BaseModel):
    """
    维度评分项
    对齐v3文档6.1节 dimensionScores 每条必传字段
    """
    dimensionCode: str = Field(
        ...,
        min_length=1,
        description="维度编码，如 SERVICE_RESPONSE",
        examples=["SERVICE_RESPONSE"],
    )
    dimensionName: str = Field(
        ...,
        min_length=1,
        description="维度名称，用于AI诊断建议生成",
        examples=["服务响应"],
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="维度得分，范围0-100",
        examples=[82.0],
    )
    avgScore: float = Field(
        ...,
        ge=0,
        le=100,
        description="该维度平均分，范围0-100，用于优势/薄弱维度判断",
        examples=[76.0],
    )


class AbnormalBehaviorItem(BaseModel):
    """
    异常行为项
    对齐v3文档6.1节 abnormalBehaviors 每条必传字段
    """
    behaviorEventId: str = Field(
        ...,
        min_length=1,
        description="行为事件ID",
        examples=["AI_BEHAVIOR_20260508103100001"],
    )
    eventTime: str = Field(
        ...,
        min_length=1,
        description="行为时间，格式yyyy-MM-dd HH:mm:ss",
        examples=["2026-05-08 10:31:00"],
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="行为摘要",
        examples=["员工未按SOP回应顾客问题"],
    )


class BehaviorStatsModel(BaseModel):
    """
    行为统计数据
    对齐v3文档6.1节 behaviorStats 必传字段
    """
    standardCount: int = Field(
        ...,
        ge=0,
        description="标准行为数",
        examples=[12],
    )
    abnormalCount: int = Field(
        ...,
        ge=0,
        description="异常行为数",
        examples=[2],
    )
    customerCount: int = Field(
        ...,
        ge=0,
        description="顾客负面行为数",
        examples=[1],
    )


class DiagnosisRequest(BaseModel):
    """
    AI时段诊断总结请求体模型
    后端 → 主网关，Content-Type: application/json
    严格对齐v3文档6.1节后端请求必传字段

    必传字段：
    - employeeNo: 员工编号（单数，单个员工）
    - startDate: 开始日期，格式 yyyy-MM-dd
    - endDate: 结束日期，格式 yyyy-MM-dd
    - score: 时间段评分
    - dimensionScores: 维度评分列表，至少1个
    - behaviorStats: 行为统计
    - abnormalBehaviors: 异常行为列表（可为空数组）
    """
    employeeNo: str = Field(
        ...,
        min_length=1,
        description="员工编号",
        examples=["EMP001"],
    )
    startDate: str = Field(
        ...,
        min_length=1,
        description="统计开始日期，格式yyyy-MM-dd",
        examples=["2026-05-02"],
    )
    endDate: str = Field(
        ...,
        min_length=1,
        description="统计结束日期，格式yyyy-MM-dd",
        examples=["2026-05-08"],
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="时间段评分，范围0-100",
        examples=[86.0],
    )
    dimensionScores: List[DimensionScoreItem] = Field(
        ...,
        min_length=1,
        description="维度评分列表，至少1个维度",
        examples=[
            [
                {"dimensionCode": "SERVICE_RESPONSE", "dimensionName": "服务响应", "score": 82, "avgScore": 76},
                {"dimensionCode": "PROFESSIONAL_SKILL", "dimensionName": "专业技能", "score": 90, "avgScore": 82},
            ]
        ],
    )
    behaviorStats: BehaviorStatsModel = Field(
        ...,
        description="行为统计数据",
        examples=[
            {
                "standardCount": 12,
                "abnormalCount": 2,
                "customerCount": 1,
            }
        ],
    )
    abnormalBehaviors: List[AbnormalBehaviorItem] = Field(
        default=[],
        description="异常行为列表（可为空数组）",
        examples=[
            [
                {
                    "behaviorEventId": "AI_BEHAVIOR_20260508103100001",
                    "eventTime": "2026-05-08 10:31:00",
                    "summary": "员工未按SOP回应顾客问题",
                }
            ]
        ],
    )


class StoreBehaviorItem(BaseModel):
    """门店诊断行为记录项"""
    behaviorEventId: str = Field(..., min_length=1, description="行为事件ID")
    behaviorType: Literal["STANDARD", "ABNORMAL", "CUSTOMER"] = Field(
        ..., description="行为类型：STANDARD/ABNORMAL/CUSTOMER"
    )
    eventTime: str = Field(..., min_length=1, description="行为时间")
    employeeId: str = Field(..., min_length=1, description="员工ID")
    employeeName: str = Field(..., min_length=1, description="员工名称")
    deviceNo: str = Field(..., min_length=1, description="设备编号")
    configItemId: str = Field(..., min_length=1, description="命中的配置项ID")
    configItemName: str = Field(..., min_length=1, description="命中的配置项名称")
    keywordContent: str = Field(..., min_length=1, description="命中的关键词内容")
    summary: str = Field(..., min_length=1, description="行为摘要")
    reviewStatus: str = Field(..., min_length=1, description="复核状态")


class StoreDiagnosisRequest(BaseModel):
    """门店AI时段诊断总结请求体"""
    storeId: str = Field(..., min_length=1, description="门店ID")
    storeName: str = Field(..., min_length=1, description="门店名称")
    startDate: str = Field(
        ..., min_length=1, description="统计开始日期，格式yyyy-MM-dd"
    )
    endDate: str = Field(
        ..., min_length=1, description="统计结束日期，格式yyyy-MM-dd"
    )
    behaviors: List[StoreBehaviorItem] = Field(
        ..., description="门店员工行为记录列表，允许为空数组"
    )


class KeywordContentItem(BaseModel):
    """词库内容项"""
    id: str = Field(..., min_length=1, description="后端生成的词库内容ID")
    content: str = Field(..., min_length=1, description="词库内容")
    matchType: Optional[str] = Field(
        None,
        description="匹配方式：sop下为FULL/SEMANTIC，forbidden/customer固定为null",
        examples=["FULL"],
    )


class KeywordConfigItem(BaseModel):
    """词库配置项分组"""
    configItemId: str = Field(..., min_length=1, description="后端生成的配置项ID")
    configItemName: str = Field(..., min_length=1, description="配置项名称")
    keywords: List[KeywordContentItem] = Field(
        ...,
        min_length=1,
        description="当前配置项下需要算法识别的词库内容",
    )


class ConfigSyncRequest(BaseModel):
    """
    词库配置同步请求体模型
    后端 → 主网关，Content-Type: application/json
    对齐交付物清单6.3节后端请求必传字段

    必传字段：
    - sop/forbidden/customer: 三类词库顶层分组，至少一类非空
    """
    sop: List[KeywordConfigItem] = Field(
        default=[],
        description="SOP话术配置，按服务场景配置项分组",
    )
    forbidden: List[KeywordConfigItem] = Field(
        default=[],
        description="违禁词配置，按违禁词类型配置项分组",
    )
    customer: List[KeywordConfigItem] = Field(
        default=[],
        description="顾客关键词配置，按顾客关键词类别配置项分组",
    )

    @model_validator(mode="after")
    def validate_at_least_one_group(self) -> "ConfigSyncRequest":
        """校验三类词库至少有一类非空。"""
        if not self.sop and not self.forbidden and not self.customer:
            raise ValueError("sop、forbidden、customer至少有一类不能为空")
        return self


# ==================== 路由器创建 ====================

backend_router = APIRouter(
    prefix="/badge/v1/algorithm",
    tags=["后端调用算法接口"],
)

# 处理器单例（由 main.py 在 lifespan 中初始化后赋值）
diagnosis_handler = DiagnosisHandler()
config_sync_handler = ConfigSyncHandler()


# ==================== 功能3：AI时段诊断总结 ====================

@backend_router.post("/users/diagnosis-summary")
async def diagnosis_summary(request: DiagnosisRequest):
    """
    响应后端AI时段诊断总结调用

    完整流程：
    1. Pydantic自动校验请求体（对齐v3文档6.1节必传字段）
    2. DiagnosisHandler进一步校验日期格式和范围
    3. 通过NodeManager负载均衡选择算力节点
    4. 将camelCase参数转换为算力节点snake_case格式后转发
    5. 同步返回诊断结果（超时30秒）

    请求体（对齐v3文档6.1节）：
    {
        "employeeNo": "EMP001",
        "startDate": "2026-05-02",
        "endDate": "2026-05-08",
        "score": 86,
        "dimensionScores": [
            {"dimensionCode": "SERVICE_RESPONSE", "dimensionName": "服务响应", "score": 86, "avgScore": 78}
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

    成功响应（算力节点原封不动返回）：
    {
        "code": 200,
        "msg": "success",
        "data": {
            "summary": "该时段员工整体表现...",
            "dimensions": [...]
        }
    }

    失败响应：
    {
        "code": 错误码,
        "msg": "错误信息",
        "data": null,
        "request_id": "..."
    }
    """
    request_id = str(uuid.uuid4())

    logger.info(
        f"收到AI诊断请求 | request_id={request_id} | "
        f"员工={request.employeeNo} | "
        f"时间范围={request.startDate} ~ {request.endDate} | "
        f"评分={request.score}"
    )

    # 将Pydantic模型转为dict传给handler（递归展开嵌套模型）
    request_dict = request.model_dump()
    result = await diagnosis_handler.handle_diagnosis(
        request_body=request_dict
    )

    return result


# ==================== 功能3.2：门店AI时段诊断总结 ====================

@backend_router.post("/stores/diagnosis-summary")
async def store_diagnosis_summary(request: StoreDiagnosisRequest):
    """
    响应后端门店AI时段诊断总结调用
    对齐交付物清单v3.5第6.2节。
    """
    request_id = str(uuid.uuid4())

    logger.info(
        f"收到门店AI诊断请求 | request_id={request_id} | "
        f"门店={request.storeId}/{request.storeName} | "
        f"时间范围={request.startDate} ~ {request.endDate} | "
        f"行为数={len(request.behaviors)}"
    )

    request_dict = request.model_dump()
    result = await diagnosis_handler.handle_store_diagnosis(
        request_body=request_dict
    )

    return result


# ==================== 功能4：词库配置同步 ====================

@backend_router.post("/config/sync")
async def config_sync(request: ConfigSyncRequest):
    """
    接收后端词库配置同步

    完整流程：
    1. Pydantic自动校验请求体（sop/forbidden/customer至少一类非空）
    2. ConfigSyncHandler校验三类词库分组
    3. 生成配置版本号，广播到所有健康算力节点
    4. 等待所有节点返回同步结果
    5. 返回同步结果汇总（全成功success=true，部分失败success=false）

    请求体（对齐v3文档6.3节）：
    {
        "sop": [
            {
                "configItemId": "scene-greeting",
                "configItemName": "迎宾接待",
                "keywords": [{"id": "sop-001", "content": "欢迎光临", "matchType": "FULL"}]
            }
        ],
        "forbidden": [],
        "customer": []
    }

    成功响应：
    {
        "code": 200,
        "msg": "ok",
        "data": {
            "success": true,
            "configVersion": "20260508103200",
            "successCount": 4,
            "failCount": 0,
            "details": [...]
        }
    }
    """
    request_id = str(uuid.uuid4())

    logger.info(
        f"收到词库配置同步请求 | request_id={request_id} | "
        f"sop={len(request.sop)} | forbidden={len(request.forbidden)} | "
        f"customer={len(request.customer)}"
    )

    # 将Pydantic模型转为dict传给handler
    result = await config_sync_handler.handle_config_sync(
        request_body=request.model_dump()
    )

    return result
