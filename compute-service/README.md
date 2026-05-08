# 智能胸牌服务管理系统 - 算力节点服务

## 项目概述

算力节点推理服务，部署在4台从树莓派（5B/8G）上，每台完整部署 ASR + LLM 推理能力。
仅与主网关通信，**绝不直接对接后端**。

## 部署架构

```
主网关(8090) ──→ 算力节点1(8091) [192.168.1.101]
              ├──→ 算力节点2(8091) [192.168.1.102]
              ├──→ 算力节点3(8091) [192.168.1.103]
              └──→ 算力节点4(8091) [192.168.1.104]
```

## 目录结构

```
compute-service/
├── main.py              # 主程序入口，FastAPI实例、生命周期管理
├── config.py            # 所有配置项集中管理（模型路径、线程数、推理参数等）
├── models.py            # 模型加载与推理（ASR + LLM，asyncio.to_thread异步包装）
├── router.py            # API路由与业务逻辑（4个接口）
├── middleware.py         # IP白名单中间件（仅允许主网关访问）
├── cache.py             # 幂等性缓存（内存版，预留Redis接口）
├── audio_utils.py       # 音频格式校验与异常音频裁剪
├── keyword_config.py    # 词库配置管理（内存+文件双缓存，实时生效）
├── exception.py         # 全局异常处理与错误码定义
├── logger.py            # loguru日志配置（控制台+文件，按天分割）
├── requirements.txt     # Python依赖清单
├── models/              # 模型文件目录（需手动放入）
│   ├── sherpa-onnx-sense-voice-zh/   # ASR模型
│   │   ├── model.onnx.int8           # 量化模型文件
│   │   └── tokens.txt                # 词表文件
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf  # LLM模型文件
├── data/                # 数据目录
│   └── keyword_config.json           # 词库配置本地缓存
└── logs/                # 日志目录（运行时自动生成）
```

## 接口清单

| 方法 | 路径 | 用途 | 调用方 |
|------|------|------|--------|
| GET | `/health` | 健康检查 | 主网关 |
| POST | `/api/v1/internal/inference/behavior-recognition` | 语音行为识别推理 | 主网关 |
| POST | `/api/v1/internal/inference/diagnosis-summary` | AI时段诊断总结推理 | 主网关 |
| POST | `/api/v1/internal/config/sync` | 词库配置同步 | 主网关 |

## 接口详细说明

### 1. 健康检查 GET /health

**无需鉴权**（但受IP白名单限制）

**响应示例：**
```json
{
  "status": "healthy",
  "node_ip": "192.168.1.101",
  "current_connections": 2,
  "config_version": "v1.0.3",
  "model_status": "loaded"
}
```

- `status`: `healthy`（模型已加载）/ `unhealthy`（模型加载失败）
- `model_status`: `loaded` / `failed`
- 模型加载失败时主网关自动摘除本节点

### 2. 语音行为识别 POST /api/v1/internal/inference/behavior-recognition

**请求格式：** FormData

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| audio_file | File | 是 | 音频文件（16000Hz、16bit、单声道WAV） |
| device_no | string | 是 | 设备编号 |
| event_time | string | 是 | 行为发生时间（yyyy-MM-dd HH:mm:ss） |
| request_id | string | 是 | 主网关生成的唯一请求ID |

**处理流程：**
1. 音频格式校验 → 不符合返回400
2. 幂等校验（5分钟内相同request_id返回缓存）
3. 并发控制（超5路返回429）
4. ASR推理 → 音频转中文文本
5. LLM推理 → 文本+词库 → 行为类型JSON
6. 异常音频裁剪（仅ABNORMAL）
7. 返回结果

**响应示例：**
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "behavior_type": "ABNORMAL",
    "summary": "员工在服务过程中使用了不当用语",
    "is_abnormal": true,
    "abnormal_audio_clip": "UklGRi4AAABXQVZFZm10..."
  },
  "request_id": "req-uuid-xxx"
}
```

### 3. AI时段诊断 POST /api/v1/internal/inference/diagnosis-summary

**请求格式：** JSON

```json
{
  "employee_no": "EMP001",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "score": 85.5,
  "dimension_scores": [
    {"dimension_code": "SERVICE_ATTITUDE", "score": 90},
    {"dimension_code": "PROFESSIONAL_SKILL", "score": 80}
  ],
  "behavior_stats": {
    "standard_count": 120,
    "abnormal_count": 5,
    "customer_count": 2
  },
  "abnormal_behaviors": [
    {"behavior_event_id": "evt-001", "event_time": "2024-01-15 14:30:00", "summary": "服务态度不佳"}
  ],
  "request_id": "req-uuid-xxx"
}
```

**响应示例：**
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "summary": "该员工本月整体服务表现良好，服务态度积极...",
    "dimensions": [
      {
        "dimension_code": "SERVICE_ATTITUDE",
        "dimension_type": "STRENGTH",
        "summary": "服务态度积极主动，获得顾客好评",
        "suggestion": ""
      },
      {
        "dimension_code": "PROFESSIONAL_SKILL",
        "dimension_type": "WEAKNESS",
        "summary": "专业技能有待提升",
        "suggestion": "建议参加服务技能培训课程"
      }
    ]
  },
  "request_id": "req-uuid-xxx"
}
```

### 4. 词库配置同步 POST /api/v1/internal/config/sync

**请求格式：** JSON

```json
{
  "config_type": "KEYWORD",
  "config_version": "v1.0.5",
  "items": [
    {"keyword": "投诉", "category": "NEGATIVE", "description": "顾客投诉相关用语"},
    {"keyword": "欢迎光临", "category": "POSITIVE", "description": "标准服务用语"}
  ]
}
```

**响应示例：**
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "success": true,
    "config_version": "v1.0.5"
  }
}
```

## 关键配置项

编辑 `config.py` 修改以下配置（部署时必须修改）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `NODE_IP` | 192.168.1.101 | 当前节点IP，每台节点不同 |
| `GATEWAY_IP_WHITELIST` | ["192.168.1.100"] | 主网关IP白名单 |
| `ASR_MODEL_DIR` | models/sherpa-onnx-sense-voice-zh | ASR模型目录 |
| `LLM_MODEL_PATH` | models/qwen2.5-1.5b-instruct-q4_k_m.gguf | LLM模型路径 |
| `MAX_CONCURRENT` | 5 | 单节点最大并发路数 |
| `ASR_NUM_THREADS` | 4 | ASR推理线程数 |
| `LLM_N_THREADS` | 4 | LLM推理线程数 |

## 树莓派部署步骤

### 1. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev
```

### 2. 创建Python虚拟环境

```bash
cd compute-service
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
# sherpa-onnx（ARM架构可能需从源码编译）
pip install sherpa-onnx

# llama-cpp-python（ARM架构需BLAS加速编译）
CMAKE_ARGS="-DGGML_BLAS=ON" pip install llama-cpp-python

# 其余依赖
pip install -r requirements.txt
```

### 4. 放置模型文件

```bash
# ASR模型（sense-voice-small）
mkdir -p models/sherpa-onnx-sense-voice-zh
# 将 model.onnx.int8 和 tokens.txt 放入上述目录

# LLM模型（qwen2.5-1.5b-instruct）
# 将 qwen2.5-1.5b-instruct-q4_k_m.gguf 放入 models/ 目录
```

### 5. 修改配置

```bash
# 编辑 config.py
# 1. 修改 NODE_IP 为当前树莓派IP
# 2. 修改 GATEWAY_IP_WHITELIST 为主网关实际IP
```

### 6. 启动服务

```bash
python main.py
# 或使用 nohup 后台运行
nohup python main.py > /dev/null 2>&1 &
```

## 枚举值（强制使用，禁止自定义）

| 枚举类 | 值 | 说明 |
|--------|-----|------|
| BehaviorType | STANDARD | 标准行为 |
| BehaviorType | ABNORMAL | 异常行为 |
| BehaviorType | CUSTOMER | 顾客负面行为 |
| DimensionType | STRENGTH | 优势维度 |
| DimensionType | WEAKNESS | 薄弱维度 |
| ConfigType | KEYWORD | 词库配置 |
| AlarmStatus | ACTIVE | 告警生效 |
| AlarmStatus | RECOVERED | 告警恢复 |

## Prompt模板修改

在 `config.py` 中修改以下变量：

- `BEHAVIOR_SYSTEM_PROMPT` - 行为识别系统提示词
- `BEHAVIOR_USER_PROMPT_TEMPLATE` - 行为识别用户提示词模板
- `DIAGNOSIS_SYSTEM_PROMPT` - 诊断总结系统提示词
- `DIAGNOSIS_USER_PROMPT_TEMPLATE` - 诊断总结用户提示词模板

## 并发与幂等机制

- **并发控制**：asyncio.Semaphore(5)，超过5路返回429
- **幂等缓存**：内存字典 + 5分钟TTL，相同request_id直接返回缓存
- **Redis扩展**：cache.py 的 get/set 接口签名与 Redis 操作一致，后续可无缝替换

## 优雅停机

关闭时自动执行：
1. 停止幂等缓存清理任务
2. 释放ASR和LLM模型资源
3. 等待现有请求处理完成（最多10秒）
4. 超时后强制关闭
