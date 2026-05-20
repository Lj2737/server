"""
智能胸牌服务管理系统 - 模型加载与推理
核心功能：
1. 服务启动时预加载ASR模型（sherpa-onnx-sense-voice-small）
2. 服务启动时初始化LLM API客户端（deepseek-v4-flash）
3. 模型加载失败时标记服务不可用
4. ASR推理：音频→中文文本（带标点）
5. LLM推理：文本→结构化JSON结果
6. 使用异步HTTP调用，避免阻塞FastAPI事件循环
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from config import (
    ASR_MODEL_DIR,
    ASR_SAMPLE_RATE,
    ASR_LANGUAGE,
    ASR_USE_ITN,
    ASR_NUM_THREADS,
    ASR_PROVIDER,
    ASR_TIMEOUT,
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_API_TIMEOUT,
    BEHAVIOR_LLM_TEMPERATURE,
    BEHAVIOR_LLM_MAX_TOKENS,
    BEHAVIOR_LLM_STOP,
    BEHAVIOR_LLM_MAX_RETRIES,
    DIAGNOSIS_LLM_TEMPERATURE,
    DIAGNOSIS_LLM_MAX_TOKENS,
    DIAGNOSIS_LLM_STOP,
    DIAGNOSIS_LLM_MAX_RETRIES,
    BEHAVIOR_SYSTEM_PROMPT,
    BEHAVIOR_USER_PROMPT_TEMPLATE,
    DIAGNOSIS_SYSTEM_PROMPT,
    DIAGNOSIS_USER_PROMPT_TEMPLATE,
    BehaviorType,
    DimensionType,
)


class ASRModel:
    """
    ASR模型封装（sherpa-onnx-sense-voice-small）
    - 启动时加载模型，避免每次请求加载
    - 提供异步推理接口，使用asyncio.to_thread包装
    - 推理超时控制：10秒
    """

    def __init__(self):
        """初始化ASR模型"""
        self._recognizer = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

    async def load(self) -> bool:
        """
        加载ASR模型
        在服务启动时调用，加载失败不影响服务启动，但健康检查会返回unhealthy

        Returns:
            是否加载成功
        """
        try:
            logger.info(f"开始加载ASR模型 | 目录={ASR_MODEL_DIR}")

            # 使用asyncio.to_thread在子线程中加载模型，避免阻塞事件循环
            self._recognizer = await asyncio.to_thread(self._load_model_sync)

            self._is_loaded = True
            logger.info("ASR模型加载成功")
            return True

        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            logger.error(f"ASR模型加载失败 | 错误={e}")
            return False

    # def _load_model_sync(self):
    #     """
    #     同步加载ASR模型（在子线程中执行）
    #     使用sherpa-onnx库加载sense-voice-small模型
    #     """
    #     try:
    #         import sherpa_onnx
    #     except ImportError:
    #         raise RuntimeError(
    #             "sherpa-onnx库未安装，请执行：pip install sherpa-onnx"
    #         )

    #     # 构建sense-voice模型配置
    #     # 【模型路径配置位置】ASR_MODEL_DIR目录下应包含model.onnx.int8和tokens.txt
    #     import os
    #     model_dir = ASR_MODEL_DIR

    #     # 查找模型文件（支持int8量化和非量化版本）
    #     model_file = None
    #     for candidate in ["model.onnx.int8", "model.onnx"]:
    #         path = os.path.join(model_dir, candidate)
    #         if os.path.exists(path):
    #             model_file = path
    #             break

    #     if model_file is None:
    #         raise FileNotFoundError(
    #             f"ASR模型文件未找到 | 目录={model_dir} | "
    #             f"需要model.onnx.int8或model.onnx"
    #         )

    #     # tokens.txt路径
    #     tokens_file = os.path.join(model_dir, "tokens.txt")
    #     if not os.path.exists(tokens_file):
    #         raise FileNotFoundError(f"tokens.txt未找到 | 路径={tokens_file}")

    #     # 创建sherpa-onnx离线识别器配置
    #     config = sherpa_onnx.OfflineRecognizerConfig(
    #         model_config=sherpa_onnx.OfflineModelConfig(
    #             sense_voice=sherpa_onnx.OfflineSenseVoiceModelConfig(
    #                 model=model_file,
    #                 language=ASR_LANGUAGE,
    #                 use_itn=ASR_USE_ITN,
    #             ),
    #             tokens=tokens_file,
    #             num_threads=ASR_NUM_THREADS,
    #             provider=ASR_PROVIDER,
    #             model_type="sense_voice",
    #         ),
    #     )

    #     # 创建识别器实例
    #     recognizer = sherpa_onnx.OfflineRecognizer(config)
    #     return recognizer
    def _load_model_sync(self):
        """同步加载ASR模型（使用新版 sherpa-onnx API）"""
        try:
            import sherpa_onnx
        except ImportError:
            raise RuntimeError("sherpa-onnx库未安装，请执行：pip install sherpa-onnx")

        import os
        model_dir = ASR_MODEL_DIR

        # 查找模型文件（支持 Q8量化版本和非量化版本）
        model_file = None
        for candidate in ["model_q8.onnx", "model.onnx"]:  # 优先用Q8量化版
            path = os.path.join(model_dir, candidate)
            if os.path.exists(path):
                model_file = path
                break

        if model_file is None:
            raise FileNotFoundError(f"ASR模型文件未找到 | 目录={model_dir}")

        # tokens.txt路径
        tokens_file = os.path.join(model_dir, "tokens.txt")
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"tokens.txt未找到 | 路径={tokens_file}")

        # 使用新版API创建识别器
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_file,
            tokens=tokens_file,
            provider=ASR_PROVIDER,
            num_threads=ASR_NUM_THREADS,
            use_itn=ASR_USE_ITN,
            sample_rate=ASR_SAMPLE_RATE,
            language=ASR_LANGUAGE,
            debug=False,
        )
        return recognizer

    @property
    def is_loaded(self) -> bool:
        """模型是否已成功加载"""
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        """模型加载失败原因"""
        return self._load_error

    async def inference(self, audio_bytes: bytes) -> str:
        """
        ASR推理：音频字节流 → 中文文本（带标点）

        Args:
            audio_bytes: WAV音频字节流（已校验格式：16000Hz、16bit、单声道）

        Returns:
            识别的中文文本（带标点）

        Raises:
            RuntimeError: ASR推理失败或输出为空
        """
        if not self._is_loaded:
            raise RuntimeError("ASR模型未加载，无法执行推理")

        try:
            # 使用asyncio.to_thread在子线程中执行推理，避免阻塞事件循环
            result_text = await asyncio.wait_for(
                asyncio.to_thread(self._inference_sync, audio_bytes),
                timeout=ASR_TIMEOUT,
            )

            # 校验输出非空
            if not result_text or not result_text.strip():
                raise RuntimeError("ASR推理输出为空")

            logger.info(f"ASR推理完成 | 输出文本长度={len(result_text)} | 文本预览={result_text[:50]}...")
            return result_text.strip()

        except asyncio.TimeoutError:
            raise RuntimeError(f"ASR推理超时（{ASR_TIMEOUT}秒）")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"ASR推理异常: {str(e)}")

    def _inference_sync(self, audio_bytes: bytes) -> str:
        """
        同步ASR推理（在子线程中执行）
        将WAV字节流转为采样数据，送入sense-voice模型推理
        """
        import numpy as np
        import wave
        import io

        # 从WAV字节流中读取采样数据
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            # 转为float32归一化数组（sherpa-onnx要求）
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # 创建推理流，送入音频数据
        stream = self._recognizer.create_stream()
        stream.accept_waveform(ASR_SAMPLE_RATE, samples)

        # 执行推理
        self._recognizer.decode_stream(stream)

        # 获取识别结果
        return stream.result.text

    def release(self) -> None:
        """释放ASR模型资源"""
        self._recognizer = None
        self._is_loaded = False
        logger.info("ASR模型资源已释放")


class LLMModel:
    """
    LLM API封装（OpenAI-compatible Chat Completions）
    - 启动时校验API配置并初始化HTTP客户端
    - 行为识别和诊断总结共用同一个客户端
    - 提供异步推理接口
    - 支持JSON输出校验，校验失败自动重试
    """

    def __init__(self):
        """初始化LLM模型"""
        self._client: Optional[httpx.AsyncClient] = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

    @staticmethod
    def _mask_secret(secret: str) -> str:
        """Return a non-sensitive preview for logs."""
        if not secret:
            return "empty"
        if len(secret) <= 10:
            return f"len={len(secret)}"
        return f"{secret[:6]}...{secret[-4:]}(len={len(secret)})"

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        """Accept either an API base URL or a full chat completions endpoint."""
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            return "https://api.deepseek.com/chat/completions"
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    async def load(self) -> bool:
        """
        初始化LLM API客户端
        在服务启动时调用

        Returns:
            是否加载成功
        """
        try:
            if not LLM_API_KEY or LLM_API_KEY.startswith("replace-with-"):
                raise RuntimeError("LLM_API_KEY未配置，请在compute-service/.env中设置")

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(LLM_API_TIMEOUT),
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            self._is_loaded = True
            self._load_error = None
            chat_url = self._chat_completions_url(LLM_API_BASE_URL)
            logger.info(
                f"LLM API客户端初始化成功 | model={LLM_MODEL_NAME} | "
                f"url={chat_url} | timeout={LLM_API_TIMEOUT}s | "
                f"key={self._mask_secret(LLM_API_KEY)}"
            )
            return True

        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            logger.error(f"LLM API客户端初始化失败 | 错误={e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """模型是否已成功加载"""
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        """模型加载失败原因"""
        return self._load_error

    async def behavior_inference(
        self, asr_text: str, keyword_config_text: str
    ) -> Dict[str, Any]:
        """
        行为识别LLM推理
        - 构造固定system_prompt和user_prompt
        - 推理参数：temperature=0.1, max_tokens=256
        - JSON输出校验，失败自动重试1次

        Args:
            asr_text: ASR转写的中文文本
            keyword_config_text: 词库配置文本

        Returns:
            解析后的JSON结果字典

        Raises:
            RuntimeError: LLM推理失败或输出格式无效
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("LLM API客户端未初始化，无法执行推理")

        # 构造prompt
        user_prompt = BEHAVIOR_USER_PROMPT_TEMPLATE.format(
            asr_text=asr_text,
            keyword_config=keyword_config_text,
        )

        # 执行推理（含重试）
        return await self._inference_with_retry(
            system_prompt=BEHAVIOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=BEHAVIOR_LLM_TEMPERATURE,
            max_tokens=BEHAVIOR_LLM_MAX_TOKENS,
            stop=BEHAVIOR_LLM_STOP,
            max_retries=BEHAVIOR_LLM_MAX_RETRIES,
            valid_keys=[
                "behavior_type",
                "summary",
                "config_item_id",
                "keyword_content",
                "is_abnormal",
            ],
        )

    async def diagnosis_inference(
        self, employee_no: str, start_date: str, end_date: str,
        score: float, dimension_scores: list, behavior_stats: dict,
        abnormal_behaviors: list,
    ) -> Dict[str, Any]:
        """
        诊断总结LLM推理
        - 构造诊断专用system_prompt和user_prompt
        - 推理参数：temperature=0.3, max_tokens=512
        - JSON输出校验，失败自动重试1次

        Args:
            employee_no: 员工编号
            start_date: 开始日期
            end_date: 结束日期
            score: 综合评分
            dimension_scores: 维度评分列表
            behavior_stats: 行为统计
            abnormal_behaviors: 异常行为列表

        Returns:
            解析后的JSON结果字典

        Raises:
            RuntimeError: LLM推理失败或输出格式无效
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("LLM API客户端未初始化，无法执行推理")

        # 构造prompt
        user_prompt = DIAGNOSIS_USER_PROMPT_TEMPLATE.format(
            employee_no=employee_no,
            start_date=start_date,
            end_date=end_date,
            score=score,
            dimension_scores=json.dumps(dimension_scores, ensure_ascii=False),
            behavior_stats=json.dumps(behavior_stats, ensure_ascii=False),
            abnormal_behaviors=json.dumps(abnormal_behaviors, ensure_ascii=False),
        )

        # 执行推理（含重试）
        return await self._inference_with_retry(
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=DIAGNOSIS_LLM_TEMPERATURE,
            max_tokens=DIAGNOSIS_LLM_MAX_TOKENS,
            stop=DIAGNOSIS_LLM_STOP,
            max_retries=DIAGNOSIS_LLM_MAX_RETRIES,
            valid_keys=["summary", "dimensions"],
        )

    async def _inference_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        stop: List[str],
        max_retries: int,
        valid_keys: List[str],
    ) -> Dict[str, Any]:
        """
        带重试的LLM推理
        - 执行推理并解析JSON输出
        - JSON校验失败时自动重试
        - 重试仍失败则返回明确错误

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数
            max_tokens: 最大输出token数
            stop: 停止标记
            max_retries: 最大重试次数
            valid_keys: 期望的JSON输出键列表（用于校验）

        Returns:
            解析后的JSON结果字典
        """
        last_error = None

        for attempt in range(1 + max_retries):
            try:
                retry_user_prompt = user_prompt
                if attempt > 0 and last_error is not None:
                    retry_user_prompt = (
                        f"{user_prompt}\n\n"
                        "The previous response was not valid JSON. "
                        "Return only one strict JSON object. "
                        "Use double quotes for every property name and string. "
                        "Do not use trailing commas, markdown, comments, or explanations."
                    )

                raw_output = await self._inference_api(
                    system_prompt,
                    retry_user_prompt,
                    temperature,
                    max_tokens,
                )

                # 解析JSON输出
                result = self._parse_json_output(raw_output)

                # 校验JSON输出是否包含必要的键
                if valid_keys:
                    for key in valid_keys:
                        if key not in result:
                            raise ValueError(f"LLM输出缺少必要字段: {key}")

                logger.info(
                    f"LLM推理成功 | 尝试次数={attempt + 1} | "
                    f"输出字段={list(result.keys())}"
                )
                return result

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                raw_preview = raw_output[:500].replace("\n", "\\n") if "raw_output" in locals() else ""
                logger.warning(
                    f"LLM输出JSON校验失败 | 尝试={attempt + 1}/{1 + max_retries} | "
                    f"error={e} | raw={raw_preview}"
                )
                if attempt < max_retries:
                    # 重试前稍等，避免立即重试得到相同结果
                    await asyncio.sleep(0.1)
                continue

            except Exception as e:
                raise RuntimeError(f"LLM推理异常: {str(e)}")

        # 所有重试均失败
        raise RuntimeError(
            f"LLM输出格式校验失败（已重试{max_retries}次）| 最后错误={last_error}"
        )

    async def _inference_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        调用OpenAI-compatible Chat Completions接口。
        """
        if self._client is None:
            raise RuntimeError("LLM API客户端未初始化")

        payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }

        # These endpoints require strict JSON in message.content. DeepSeek's
        # thinking mode can return an empty content field or non-JSON reasoning
        # text, so keep structured output calls in non-thinking mode.

        chat_url = self._chat_completions_url(LLM_API_BASE_URL)
        response = await self._client.post(chat_url, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 401:
                raise RuntimeError(
                    "LLM API authentication failed (401). Check compute-service/.env "
                    "LLM_API_KEY/DEEPSEEK_API_KEY and model access; "
                    f"current key={self._mask_secret(LLM_API_KEY)}, url={chat_url}"
                ) from e
            raise
        result = response.json()
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not content.strip():
            response_preview = json.dumps(result, ensure_ascii=False)[:800]
            for fallback_value in (
                message.get("reasoning_content"),
                message.get("reasoning"),
                message.get("output_text"),
                message.get("text"),
                choice.get("text"),
            ):
                if isinstance(fallback_value, str) and "{" in fallback_value:
                    json_fallback = self._extract_json_object(fallback_value)
                    if json_fallback and json_fallback.strip().startswith("{"):
                        return json_fallback.strip()
            logger.warning(
                "LLM API returned empty message.content | "
                f"finish_reason={choice.get('finish_reason')} | "
                f"message_keys={list(message.keys())} | response={response_preview}"
            )
            raise RuntimeError(
                "LLM API returned empty message.content. Structured JSON calls "
                "must run without thinking/reasoning output."
            )
        return content.strip()

    @staticmethod
    def _parse_json_output(raw_output: str) -> Dict[str, Any]:
        """
        解析LLM输出的JSON文本
        - 尝试直接解析
        - 如果包含Markdown代码块标记，提取后再解析
        - 清理可能的前后空白和换行

        Args:
            raw_output: LLM原始输出文本

        Returns:
            解析后的字典

        Raises:
            json.JSONDecodeError: JSON解析失败
        """
        text = raw_output.strip()

        # 清理Markdown代码块标记（LLM可能输出```json...```）
        if text.startswith("```"):
            # 移除开头的```json或```
            lines = text.split("\n")
            # 移除第一行（```json或```）
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 移除最后一行（```）
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        candidates = [text]
        extracted = LLMModel._extract_json_object(text)
        if extracted and extracted != text:
            candidates.append(extracted)

        last_error: Optional[json.JSONDecodeError] = None
        for candidate in candidates:
            for cleaned in (candidate, LLMModel._sanitize_json_text(candidate)):
                try:
                    parsed = json.loads(cleaned)
                    if not isinstance(parsed, dict):
                        raise ValueError("LLM output JSON root must be an object")
                    return parsed
                except json.JSONDecodeError as e:
                    last_error = e

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("No JSON object found", text, 0)

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            return text

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return text[start:]

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.replace("\ufeff", "")
        cleaned = cleaned.translate({0x201C: 34, 0x201D: 34, 0x2018: 39, 0x2019: 39})
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        cleaned = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', cleaned)
        return cleaned

    def release(self) -> None:
        """释放LLM API客户端引用。"""
        if self._client is not None:
            try:
                if not self._client.is_closed:
                    asyncio.create_task(self._client.aclose())
            except RuntimeError:
                pass
            self._client = None
        self._is_loaded = False
        logger.info("LLM API客户端已释放")


# ==================== 全局模型实例 ====================

# 全局ASR模型实例（单例，服务启动时加载）
asr_model = ASRModel()

# 全局LLM模型实例（单例，行为识别和诊断总结共用）
llm_model = LLMModel()


def get_model_status() -> str:
    """
    获取模型加载状态
    - 两个模型均加载成功：返回"loaded"
    - 任一模型加载失败：返回"failed"

    Returns:
        模型状态字符串：loaded/failed
    """
    if asr_model.is_loaded and llm_model.is_loaded:
        return "loaded"
    return "failed"
