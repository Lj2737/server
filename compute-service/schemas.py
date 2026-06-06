"""
智能胸牌服务管理系统 - 请求/响应Pydantic模型定义
核心作用：
1. FastAPI自动根据模型生成Swagger UI文档（参数、类型、必填、示例值一目了然）
2. FastAPI自动校验请求体（类型、必填、格式、枚举值），不合法直接422拒绝
3. 消除router.py中冗长的手动校验代码
4. 响应模型约束输出格式，保证接口契约一致性
"""
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ==================== 通用响应模型 ====================

class ApiResponse(BaseModel):
    """统一API响应格式"""
    code: int = Field(..., description="业务状态码，200=成功")
    msg: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    request_id: str = Field("", description="请求ID")


# ==================== 诊断总结接口 ====================

class DiagnosisSummaryRequest(BaseModel):
    """
    AI时段诊断总结推理请求体
    所有字段为必填，由主网关组装后转发
    """
    employee_no: str = Field(
        ...,
        min_length=1,
        description="员工编号",
        examples=["EMP001"],
    )
    start_date: str = Field(
        ...,
        min_length=1,
        description="统计开始日期，格式yyyy-MM-dd",
        examples=["2026-05-01"],
    )
    end_date: str = Field(
        ...,
        min_length=1,
        description="统计结束日期，格式yyyy-MM-dd",
        examples=["2026-05-07"],
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="综合评分，范围0-100",
        examples=[85.5],
    )
    dimension_scores: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="维度评分列表，至少1个维度",
        examples=[
            [
                {"dimension_code": "SERVICE_ATTITUDE", "score": 90.0, "avg_score": 82.0},
                {"dimension_code": "PROFESSIONAL_SKILL", "score": 80.0, "avg_score": 74.0},
            ]
        ],
    )
    behavior_stats: Dict[str, Any] = Field(
        ...,
        description="行为统计数据",
        examples=[
            {
                "total_count": 120,
                "standard_count": 100,
                "abnormal_count": 15,
                "customer_count": 5,
            }
        ],
    )
    abnormal_behaviors: List[Dict[str, Any]] = Field(
        ...,
        description="异常行为列表（可为空列表）",
        examples=[
            [
                {
                    "behavior_type": "ABNORMAL",
                    "event_time": "2026-05-03 14:30:00",
                    "summary": "服务态度不佳",
                }
            ]
        ],
    )
    request_id: str = Field(
        ...,
        min_length=1,
        description="主网关生成的唯一请求ID（幂等键）",
        examples=["req-diag-001"],
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """校验日期格式必须为yyyy-MM-dd"""
        try:
            datetime.datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"日期格式错误，要求yyyy-MM-dd，实际={v}")
        return v


class StoreBehaviorItem(BaseModel):
    """门店诊断行为记录项"""
    behavior_event_id: str = Field(..., min_length=1, description="行为事件ID")
    behavior_type: Literal["STANDARD", "ABNORMAL", "CUSTOMER"] = Field(
        ..., description="行为类型：STANDARD/ABNORMAL/CUSTOMER"
    )
    event_time: str = Field(..., min_length=1, description="行为时间")
    employee_id: str = Field(..., min_length=1, description="员工ID")
    employee_name: str = Field(..., min_length=1, description="员工名称")
    device_no: str = Field(..., min_length=1, description="设备编号")
    config_item_id: str = Field(..., min_length=1, description="命中的配置项ID")
    config_item_name: str = Field(..., min_length=1, description="命中的配置项名称")
    keyword_content: str = Field(..., min_length=1, description="命中的关键词内容")
    summary: str = Field(..., min_length=1, description="行为摘要")
    review_status: str = Field(..., min_length=1, description="复核状态")


class StoreDiagnosisSummaryRequest(BaseModel):
    """门店AI时段诊断总结推理请求体"""
    store_id: str = Field(..., min_length=1, description="门店ID")
    store_name: str = Field(..., min_length=1, description="门店名称")
    start_date: str = Field(..., min_length=1, description="开始日期，格式yyyy-MM-dd")
    end_date: str = Field(..., min_length=1, description="结束日期，格式yyyy-MM-dd")
    behaviors: List[StoreBehaviorItem] = Field(
        ..., description="门店员工行为记录列表，允许为空数组"
    )
    request_id: str = Field(..., min_length=1, description="请求ID")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """校验日期格式必须为yyyy-MM-dd"""
        try:
            datetime.datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"日期格式错误，要求yyyy-MM-dd，实际={v}")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "StoreDiagnosisSummaryRequest":
        """校验结束日期不能早于开始日期"""
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        if end < start:
            raise ValueError("end_date不能早于start_date")
        return self


class DimensionItem(BaseModel):
    """诊断维度项"""
    dimension_code: str = Field(..., description="维度编码")
    dimension_type: str = Field(
        ...,
        description="维度类型枚举：STRENGTH（优势）/ WEAKNESS（薄弱）",
    )
    summary: str = Field(..., description="该维度表现总结，100字以内")
    suggestion: Optional[str] = Field(
        None,
        description="改进建议（仅dimension_type=WEAKNESS时必填，100字以内）",
    )


class DiagnosisResultData(BaseModel):
    """诊断总结结果数据"""
    summary: str = Field(..., description="该时间段总体服务表现总结，150字以内")
    dimensions: List[DimensionItem] = Field(
        ..., description="多维度诊断结果列表"
    )


class DiagnosisSummaryResponse(BaseModel):
    """AI时段诊断总结推理响应"""
    code: int = Field(200, description="业务状态码")
    msg: str = Field("success", description="响应消息")
    data: DiagnosisResultData = Field(..., description="诊断结果数据")
    request_id: str = Field(..., description="请求ID")


class StoreDiagnosisResultData(BaseModel):
    """门店诊断总结结果数据"""
    summary: str = Field(..., description="门店时段综合分析")
    suggestions: List[str] = Field(..., description="门店分析建议")


class StoreDiagnosisSummaryResponse(BaseModel):
    """门店AI时段诊断总结推理响应"""
    code: int = Field(200, description="业务状态码")
    msg: str = Field("success", description="响应消息")
    data: StoreDiagnosisResultData = Field(..., description="门店诊断结果数据")
    request_id: str = Field(..., description="请求ID")


# ==================== 词库配置同步接口 ====================

class KeywordItem(BaseModel):
    """词库配置项"""
    keyword: str = Field(..., min_length=1, description="关键词")
    category: str = Field("", description="分类标签")
    weight: float = Field(1.0, ge=0, description="权重，默认1.0")


class KeywordContentItem(BaseModel):
    """词库内容项，对齐后端同步给算法的keywords结构"""
    id: str = Field(..., min_length=1, description="后端生成的词库内容ID")
    content: str = Field(..., min_length=1, description="词库内容")
    matchType: Optional[str] = Field(
        None,
        description="匹配方式：sop下为FULL/SEMANTIC，forbidden/customer固定为null",
    )

    @field_validator("matchType")
    @classmethod
    def validate_match_type(cls, v: Optional[str]) -> Optional[str]:
        """校验matchType有值时只能为FULL或SEMANTIC"""
        if v is not None and v not in {"FULL", "SEMANTIC"}:
            raise ValueError(f"matchType无效：期望FULL/SEMANTIC/null，实际={v}")
        return v


class KeywordConfigItem(BaseModel):
    """配置项分组，包含配置项ID、名称及其下关键词"""
    configItemId: str = Field(..., min_length=1, description="后端生成的配置项ID")
    configItemName: str = Field(..., min_length=1, description="配置项名称")
    keywords: List[KeywordContentItem] = Field(
        ...,
        min_length=1,
        description="当前配置项下需要算法识别的词库内容",
    )


class ConfigSyncRequest(BaseModel):
    """
    词库配置同步请求体
    主网关广播词库时使用
    """
    config_type: str = Field(
        ...,
        description="配置类型枚举，当前仅支持KEYWORD",
        examples=["KEYWORD"],
    )
    config_version: str = Field(
        ...,
        min_length=1,
        description="配置版本号，递增字符串",
        examples=["v1.0.0"],
    )
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

    @field_validator("config_type")
    @classmethod
    def validate_config_type(cls, v: str) -> str:
        """校验配置类型必须为KEYWORD"""
        if v != "KEYWORD":
            raise ValueError(f"配置类型无效：期望KEYWORD，实际={v}")
        return v

    @model_validator(mode="after")
    def validate_config_data(self) -> "ConfigSyncRequest":
        """校验三类词库至少有一类非空"""
        if not self.sop and not self.forbidden and not self.customer:
            raise ValueError("sop、forbidden、customer至少有一类不能为空")
        return self


class ConfigSyncResultData(BaseModel):
    """配置同步结果数据"""
    success: bool = Field(..., description="是否同步成功")
    config_version: str = Field(..., description="当前生效的配置版本号")


class ConfigSyncResponse(BaseModel):
    """词库配置同步响应"""
    code: int = Field(200, description="业务状态码")
    msg: str = Field("success", description="响应消息")
    data: ConfigSyncResultData = Field(..., description="同步结果数据")


# ==================== 健康检查响应 ====================

class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="节点状态：healthy / unhealthy")
    node_ip: str = Field(..., description="节点IP地址")
    current_connections: int = Field(
        ..., description="当前活跃连接数"
    )
    config_version: str = Field(
        ..., description="当前词库配置版本号"
    )
    model_status: str = Field(
        ..., description="模型加载状态：loaded / failed"
    )


# ==================== 行为识别响应 ====================

class BehaviorResultData(BaseModel):
    """行为识别结果数据"""
    behavior_type: str = Field(
        ...,
        description="行为类型枚举：STANDARD / ABNORMAL / CUSTOMER",
    )
    summary: str = Field(..., description="行为摘要，100字以内")
    config_item_id: str = Field(
        "",
        description="命中的配置项ID，回调后端时映射为configItemId",
    )
    keyword_content: str = Field(
        "",
        description="命中的关键词内容，回调后端时映射为keywordContent",
    )
    keyword_matches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="命中的全部关键词列表，包含config_item_id、keyword_content、behavior_type等字段",
    )
    is_abnormal: bool = Field(..., description="是否异常行为")
    abnormal_audio_clip: Optional[str] = Field(
        None,
        description="异常音频片段Base64（仅behavior_type=ABNORMAL时返回）",
    )


class BehaviorRecognitionResponse(BaseModel):
    """语音行为识别推理响应"""
    code: int = Field(200, description="业务状态码")
    msg: str = Field("success", description="响应消息")
    data: BehaviorResultData = Field(..., description="行为识别结果数据")
    request_id: str = Field(..., description="请求ID")
