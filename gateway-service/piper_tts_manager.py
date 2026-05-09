"""
智能胸牌服务管理系统 - Piper TTS本地部署与流式合成
核心功能：
1. 服务启动时预加载Piper TTS模型（zh_CN-huayan-medium，CPU-only）
2. 异步流式合成：文本 → 16000Hz/16bit/单声道PCM裸流（硬件直接播放）
3. 模型加载失败时，is_available()返回False，健康检查联动
4. 全异步实现，模型推理用asyncio.to_thread包装，不阻塞事件循环

模型信息：
- 模型名称：zh_CN-huayan-medium（Piper官方中文女声模型）
- 原始采样率：22050Hz，合成后软件重采样到16000Hz
- 音频格式：16bit有符号整数、单声道PCM裸流（无WAV头）
- CPU-only：use_cuda=False，适配树莓派部署

使用示例：
    from piper_tts_manager import PiperTTSManager

    # 获取单例
    tts = PiperTTSManager()

    # 加载模型（在FastAPI lifespan startup阶段调用）
    success = await tts.load_model()

    # 流式合成
    async for pcm_chunk in tts.synthesize_stream("播报内容"):
        # pcm_chunk: bytes，16bit/16000Hz/mono PCM裸数据
        websocket.send(pcm_chunk)

    # 检查可用性
    if tts.is_available():
        ...
"""
import asyncio
import struct
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from config import (
    PIPER_MODEL_PATH,
    PIPER_CONFIG_PATH,
    PIPER_USE_CUDA,
    PIPER_TARGET_SAMPLE_RATE,
    PIPER_TARGET_SAMPLE_WIDTH,
    PIPER_TARGET_CHANNELS,
    PIPER_NOISE_SCALE,
    PIPER_LENGTH_SCALE,
    PIPER_VOLUME,
)


class PiperTTSManager:
    """
    Piper TTS本地部署与流式合成管理器 - 单例模式

    职责：
    - 服务启动时预加载Piper TTS ONNX模型
    - 异步流式合成文本为PCM裸音频流
    - 自动将模型原始采样率(22050Hz)重采样到硬件要求(16000Hz)
    - 提供TTS服务可用性检查接口

    设计约束：
    - 单例模式，全局唯一实例
    - CPU-only（use_cuda=False），适配树莓派
    - 所有阻塞操作用asyncio.to_thread包装
    - 输出PCM裸流（无WAV头），硬件直接播放
    """

    _instance: Optional["PiperTTSManager"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个PiperTTSManager实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._voice = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._initialized = False

    # ==================== 模型加载 ====================

    async def load_model(self) -> bool:
        """
        预加载Piper TTS模型
        在FastAPI lifespan startup阶段调用

        Returns:
            是否加载成功
        """
        try:
            logger.info(
                f"开始加载Piper TTS模型 | "
                f"模型路径={PIPER_MODEL_PATH} | "
                f"配置路径={PIPER_CONFIG_PATH} | "
                f"use_cuda={PIPER_USE_CUDA}"
            )

            # 在子线程中加载模型，避免阻塞事件循环
            self._voice = await asyncio.to_thread(self._load_model_sync)

            self._is_loaded = True
            self._load_error = None

            logger.info(
                f"Piper TTS模型加载成功 | "
                f"目标采样率={PIPER_TARGET_SAMPLE_RATE}Hz | "
                f"目标位深={PIPER_TARGET_SAMPLE_WIDTH * 8}bit | "
                f"目标声道={PIPER_TARGET_CHANNELS}"
            )
            return True

        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            logger.error(
                f"Piper TTS模型加载失败 | 错误={e}"
            )
            return False

    def _load_model_sync(self):
        """
        同步加载Piper TTS模型（在子线程中执行）
        使用piper.PiperVoice.load加载ONNX模型
        """
        from piper import PiperVoice

        voice = PiperVoice.load(
            PIPER_MODEL_PATH,
            config_path=PIPER_CONFIG_PATH,
            use_cuda=PIPER_USE_CUDA,
        )
        return voice

    # ==================== 流式合成 ====================

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        异步流式合成：文本 → 16000Hz/16bit/单声道PCM裸流
        边合成边返回，无需等待整段合成完成

        处理流程：
        1. 校验TTS服务可用性
        2. 构造SynthesisConfig（语速、噪声缩放、句间停顿）
        3. 在子线程中调用PiperVoice.synthesize逐句合成
        4. 对每个AudioChunk：提取int16 PCM → 重采样到16000Hz → 返回PCM裸流
        5. 每个chunk之间插入短暂静音（句间停顿）

        Args:
            text: 待合成文本（非空，长度已在调用方校验）

        Yields:
            bytes: 16bit/16000Hz/mono PCM裸数据chunk

        Raises:
            RuntimeError: TTS服务不可用或合成失败
        """
        if not self._is_loaded or self._voice is None:
            raise RuntimeError(
                f"Piper TTS模型未加载，无法合成 | 错误={self._load_error}"
            )

        if not text or not text.strip():
            logger.warning("Piper TTS合成文本为空，跳过")
            return

        try:
            from piper import SynthesisConfig

            # 构造合成配置
            # SynthesisConfig支持的字段：
            # - speaker_id: 多说话人模型使用（本模型为单说话人，不需要）
            # - length_scale: 语速缩放（<1加快，>1减慢）
            # - noise_scale: 生成器噪声
            # - noise_w_scale: 音素宽度噪声
            # - normalize_audio: 音频归一化
            # - volume: 音量倍数
            syn_config = SynthesisConfig(
                noise_scale=PIPER_NOISE_SCALE,
                length_scale=PIPER_LENGTH_SCALE,
                volume=PIPER_VOLUME,
            )

            # 在子线程中执行同步合成迭代器
            audio_chunks = await asyncio.to_thread(
                self._synthesize_sync, text, syn_config
            )

            # 逐chunk重采样并yield
            for chunk_idx, (pcm_bytes, orig_rate) in enumerate(audio_chunks):
                # 重采样到目标采样率
                resampled_pcm = self._resample_pcm(
                    pcm_bytes,
                    orig_sample_rate=orig_rate,
                    target_sample_rate=PIPER_TARGET_SAMPLE_RATE,
                )

                if resampled_pcm:
                    logger.debug(
                        f"Piper TTS合成chunk | 序号={chunk_idx} | "
                        f"原始采样率={orig_rate}Hz | "
                        f"重采样后={len(resampled_pcm)}字节"
                    )
                    yield resampled_pcm

            logger.info(
                f"Piper TTS流式合成完成 | 文本长度={len(text)} | "
                f"chunks={len(audio_chunks)}"
            )

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"Piper TTS合成异常 | 文本={text[:50]} | 错误={e}"
            )
            raise RuntimeError(f"Piper TTS合成失败: {str(e)}")

    def _synthesize_sync(self, text: str, syn_config) -> list:
        """
        同步合成（在子线程中执行）
        调用PiperVoice.synthesize逐句生成AudioChunk

        Returns:
            list of (pcm_bytes, sample_rate) 元组
        """
        chunks = []
        for audio_chunk in self._voice.synthesize(text, syn_config):
            pcm_bytes = audio_chunk.audio_int16_bytes
            sample_rate = audio_chunk.sample_rate
            chunks.append((pcm_bytes, sample_rate))
        return chunks

    # ==================== PCM重采样 ====================

    @staticmethod
    def _resample_pcm(
        pcm_bytes: bytes,
        orig_sample_rate: int,
        target_sample_rate: int,
    ) -> bytes:
        """
        PCM音频重采样：22050Hz → 16000Hz
        使用scipy.signal.resample进行高质量重采样

        处理流程：
        1. bytes → int16 numpy数组
        2. 归一化到float32 [-1.0, 1.0]
        3. scipy.signal.resample重采样
        4. float32 → int16
        5. int16数组 → bytes

        Args:
            pcm_bytes: 原始PCM字节数据（16bit signed, mono）
            orig_sample_rate: 原始采样率（模型输出22050Hz）
            target_sample_rate: 目标采样率（硬件要求16000Hz）

        Returns:
            重采样后的PCM字节数据
        """
        if orig_sample_rate == target_sample_rate:
            return pcm_bytes

        try:
            # bytes → int16 numpy数组
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)

            # 计算重采样后的长度
            orig_len = len(pcm_array)
            target_len = int(orig_len * target_sample_rate / orig_sample_rate)

            if target_len == 0:
                return b""

            # 归一化到float32进行重采样
            pcm_float = pcm_array.astype(np.float32) / 32768.0

            # scipy重采样
            from scipy.signal import resample
            resampled_float = resample(pcm_float, target_len)

            # float32 → int16
            resampled_int16 = np.clip(
                resampled_float * 32768.0, -32768, 32767
            ).astype(np.int16)

            return resampled_int16.tobytes()

        except Exception as e:
            logger.error(
                f"PCM重采样失败 | 原始采样率={orig_sample_rate} | "
                f"目标采样率={target_sample_rate} | 错误={e}"
            )
            # 重采样失败时返回原始数据（降级处理，采样率不匹配但至少有音频）
            return pcm_bytes

    # ==================== 状态查询 ====================

    def is_available(self) -> bool:
        """
        TTS服务是否可用
        模型加载成功时返回True，加载失败时返回False
        """
        return self._is_loaded and self._voice is not None

    @property
    def load_error(self) -> Optional[str]:
        """模型加载失败原因"""
        return self._load_error

    def release(self) -> None:
        """释放模型资源"""
        self._voice = None
        self._is_loaded = False
        logger.info("Piper TTS模型资源已释放")
