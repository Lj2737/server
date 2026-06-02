"""
智能胸牌服务管理系统 - 音频校验与裁剪工具
核心功能：
1. WAV音频格式校验：16000Hz采样率、16bit位深（支持多声道自动转单声道）
2. 多声道自动转单声道：双声道/立体声输入时自动降混为单声道
3. 异常音频片段裁剪：触发前5秒+触发后10秒，共15秒
4. 裁剪后音频Base64编码
"""
import wave
import io
import base64
import struct
import re
from typing import Tuple, Optional, List, Any

from loguru import logger

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH,
    AUDIO_CHANNELS,
    AUDIO_MAX_CHANNELS,
    ABNORMAL_CLIP_BEFORE_SECONDS,
    ABNORMAL_CLIP_AFTER_SECONDS,
    ABNORMAL_CLIP_TOTAL_SECONDS,
)


def _convert_to_mono(audio_bytes: bytes) -> bytes:
    """
    将多声道WAV音频转换为单声道WAV音频
    降混方式：取所有声道的平均值（标准音频降混算法）

    Args:
        audio_bytes: 原始WAV音频字节流（可能为多声道）

    Returns:
        单声道WAV音频字节流（原始就是单声道时直接返回）
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()

            # 已经是单声道，无需转换
            if channels == 1:
                return audio_bytes

            logger.info(
                f"多声道自动转单声道 | 原始声道数={channels} | "
                f"采样率={frame_rate}Hz | 位深={sample_width * 8}bit | "
                f"帧数={n_frames}"
            )

            # 读取所有帧数据
            frames_data = wf.readframes(n_frames)

            # 按声道解析并降混
            mono_frames = _downmix_channels(
                frames_data, channels, sample_width, n_frames
            )

            # 创建单声道WAV
            output_buffer = io.BytesIO()
            with wave.open(output_buffer, "wb") as out_wf:
                out_wf.setnchannels(1)
                out_wf.setsampwidth(sample_width)
                out_wf.setframerate(frame_rate)
                out_wf.writeframes(mono_frames)

            result = output_buffer.getvalue()
            logger.info(
                f"声道转换完成 | 原始大小={len(audio_bytes)}字节 | "
                f"转换后大小={len(result)}字节"
            )
            return result

    except Exception as e:
        logger.exception(f"声道转换失败，返回原始音频 | 错误={e}")
        return audio_bytes


def _downmix_channels(
    frames_data: bytes,
    channels: int,
    sample_width: int,
    n_frames: int,
) -> bytes:
    """
    多声道降混为单声道
    降混算法：取所有声道采样值的算术平均值，再截断到合法范围

    支持的位深：
    - 16bit (sample_width=2)：有符号整数，范围[-32768, 32767]

    Args:
        frames_data: 原始帧数据
        channels: 声道数
        sample_width: 采样位深（字节数）
        n_frames: 帧数

    Returns:
        单声道帧数据
    """
    mono_samples = []

    if sample_width == 2:
        # 16bit：每个采样2字节，有符号短整型
        fmt = f"<{n_frames * channels}h"
        try:
            all_samples = struct.unpack(fmt, frames_data)
        except struct.error:
            # 帧数据可能不完整，按实际长度解析
            actual_samples = len(frames_data) // sample_width
            fmt = f"<{actual_samples}h"
            all_samples = struct.unpack(fmt, frames_data[:actual_samples * sample_width])

        # 每 channels 个采样取平均值
        for i in range(0, len(all_samples), channels):
            frame_samples = all_samples[i:i + channels]
            avg = sum(frame_samples) // len(frame_samples)
            # 截断到16bit范围
            avg = max(-32768, min(32767, avg))
            mono_samples.append(avg)

        return struct.pack(f"<{len(mono_samples)}h", *mono_samples)

    else:
        # 其他位深暂不支持降混，返回静音
        logger.warning(f"不支持的位深进行降混：{sample_width * 8}bit，返回静音")
        return b"\x00\x00" * n_frames


def validate_wav_audio(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    校验WAV音频格式
    强制要求：16000Hz采样率、16bit位深
    声道：支持单声道和多声道输入，多声道时自动转换为单声道
    直接适配sense-voice模型输入要求（模型仅需单声道）

    Args:
        audio_bytes: 音频文件的原始字节流

    Returns:
        (是否通过校验, 校验结果描述)
    """
    if not audio_bytes or len(audio_bytes) < 44:
        # WAV文件头最少44字节
        return False, "音频文件为空或过小，无法解析WAV头"

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()

            # 校验采样位深：必须为16bit（2字节）
            if sample_width != AUDIO_SAMPLE_WIDTH:
                return False, (
                    f"音频位深不符：要求{AUDIO_SAMPLE_WIDTH * 8}bit，"
                    f"实际{sample_width * 8}bit"
                )

            # 校验采样率：必须为16000Hz
            if frame_rate != AUDIO_SAMPLE_RATE:
                return False, (
                    f"音频采样率不符：要求{AUDIO_SAMPLE_RATE}Hz，"
                    f"实际{frame_rate}Hz"
                )

            # 校验声道数：允许1到AUDIO_MAX_CHANNELS声道，多声道时自动转换
            if channels < 1 or channels > AUDIO_MAX_CHANNELS:
                return False, (
                    f"音频声道数不支持：要求1-{AUDIO_MAX_CHANNELS}声道，"
                    f"实际{channels}声道"
                )

            # 校验音频时长：至少0.1秒，避免空音频或极短音频
            duration = n_frames / frame_rate
            if duration < 0.1:
                return False, f"音频时长过短：{duration:.2f}秒，至少需要0.1秒"

            # 校验音频时长：不超过5分钟，避免超长音频消耗资源
            if duration > 300:
                return False, f"音频时长过长：{duration:.2f}秒，最大支持300秒"

            # 多声道提示
            channel_info = ""
            if channels > 1:
                channel_info = f"（{channels}声道将自动转换为单声道）"

            return True, (
                f"音频格式校验通过 | 时长={duration:.2f}秒 | "
                f"声道数={channels}{channel_info}"
            )

    except wave.Error as e:
        return False, f"音频文件解析失败，非标准WAV格式: {str(e)}"
    except Exception as e:
        return False, f"音频校验异常: {str(e)}"


def ensure_mono_audio(audio_bytes: bytes) -> bytes:
    """
    确保音频为单声道格式
    如果输入是多声道，自动降混为单声道
    如果输入已经是单声道，直接返回原始数据

    在validate_wav_audio校验通过后调用此函数，
    确保后续ASR推理和音频裁剪使用统一的单声道格式。

    Args:
        audio_bytes: WAV音频字节流（已通过格式校验）

    Returns:
        单声道WAV音频字节流
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            if channels == AUDIO_CHANNELS:
                # 已经是单声道，直接返回
                return audio_bytes
    except Exception:
        pass

    # 多声道，执行转换
    return _convert_to_mono(audio_bytes)


def clip_abnormal_audio(
    audio_bytes: bytes,
    asr_text: str = "",
    keyword_content: str = "",
    asr_tokens: Optional[List[Any]] = None,
    asr_timestamps: Optional[List[Any]] = None,
) -> str:
    """
    裁剪异常行为音频片段
    - 仅在behavior_type=ABNORMAL时调用
    - 从原始音频中裁剪：触发前5秒 + 触发后10秒 = 共15秒
    - 裁剪逻辑：以音频末尾为触发时刻，往前取15秒
    - 裁剪后格式：16000Hz、16bit、单声道WAV
    - 输出：Base64编码字符串

    Args:
        audio_bytes: 原始WAV音频字节流（应已通过ensure_mono_audio处理）

    Returns:
        Base64编码的裁剪后WAV音频，异常时返回空字符串
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frame_rate = wf.getframerate()
            total_frames = wf.getnframes()
            total_duration = total_frames / frame_rate

            trigger_seconds, trigger_method = _estimate_trigger_seconds(
                total_duration=total_duration,
                asr_text=asr_text,
                keyword_content=keyword_content,
                asr_tokens=asr_tokens,
                asr_timestamps=asr_timestamps,
            )

            if total_duration <= ABNORMAL_CLIP_TOTAL_SECONDS:
                # 音频短于15秒，取全部音频
                start_frame = 0
                read_frames = total_frames
                logger.debug(
                    f"音频短于裁剪长度，取全部 | "
                    f"音频时长={total_duration:.2f}s | 裁剪目标={ABNORMAL_CLIP_TOTAL_SECONDS}s"
                )
            else:
                start_seconds = max(0.0, trigger_seconds - ABNORMAL_CLIP_BEFORE_SECONDS)
                end_seconds = min(total_duration, trigger_seconds + ABNORMAL_CLIP_AFTER_SECONDS)

                if end_seconds - start_seconds < ABNORMAL_CLIP_TOTAL_SECONDS:
                    if start_seconds <= 0:
                        end_seconds = min(total_duration, ABNORMAL_CLIP_TOTAL_SECONDS)
                    elif end_seconds >= total_duration:
                        start_seconds = max(0.0, total_duration - ABNORMAL_CLIP_TOTAL_SECONDS)

                start_frame = int(start_seconds * frame_rate)
                end_frame = min(total_frames, int(end_seconds * frame_rate))
                read_frames = max(0, end_frame - start_frame)
                logger.debug(
                    f"异常音频按关键词位置裁剪 | keyword={keyword_content} | "
                    f"method={trigger_method} | trigger={trigger_seconds:.2f}s | "
                    f"start={start_seconds:.2f}s | "
                    f"end={end_seconds:.2f}s | 起始帧={start_frame} | "
                    f"裁剪帧数={read_frames} | 音频时长={total_duration:.2f}s"
                )

            # 定位到起始帧并读取
            wf.setpos(start_frame)
            frames_data = wf.readframes(read_frames)

            if not frames_data:
                logger.warning(
                    f"异常音频裁剪结果为空，回退为整段音频 | "
                    f"音频时长={total_duration:.2f}s | keyword={keyword_content}"
                )
                wf.setpos(0)
                frames_data = wf.readframes(total_frames)
                read_frames = total_frames

        # 创建新的WAV字节流（确保格式：16000Hz、16bit、单声道）
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, "wb") as out_wf:
            out_wf.setnchannels(AUDIO_CHANNELS)        # 单声道
            out_wf.setsampwidth(AUDIO_SAMPLE_WIDTH)      # 16bit
            out_wf.setframerate(AUDIO_SAMPLE_RATE)       # 16000Hz
            out_wf.writeframes(frames_data)

        # Base64编码
        wav_bytes = output_buffer.getvalue()
        base64_str = base64.b64encode(wav_bytes).decode("utf-8")

        logger.info(
            f"异常音频裁剪完成 | 原始时长={total_duration:.2f}s | "
            f"裁剪后={read_frames / frame_rate:.2f}s | "
            f"Base64大小={len(base64_str)}字符"
        )
        return base64_str

    except Exception as e:
        logger.exception(f"异常音频裁剪失败 | 错误={e}")
        return ""


def _estimate_trigger_seconds(
    total_duration: float,
    asr_text: str = "",
    keyword_content: str = "",
    asr_tokens: Optional[List[Any]] = None,
    asr_timestamps: Optional[List[Any]] = None,
) -> Tuple[float, str]:
    timestamp_trigger = _estimate_trigger_seconds_from_timestamps(
        total_duration=total_duration,
        keyword_content=keyword_content,
        asr_tokens=asr_tokens,
        asr_timestamps=asr_timestamps,
    )
    if timestamp_trigger is not None:
        return timestamp_trigger, "timestamp"

    source_text = (asr_text or "").strip()
    keyword = (keyword_content or "").strip()
    if not source_text or not keyword:
        return total_duration, "fallback_end_missing_text_or_keyword"

    keyword_index = source_text.find(keyword)
    if keyword_index < 0:
        return total_duration, "fallback_end_keyword_not_found"

    midpoint = keyword_index + len(keyword) / 2
    ratio = midpoint / max(len(source_text), 1)
    ratio = max(0.0, min(1.0, ratio))
    return total_duration * ratio, "text_ratio"


def _estimate_trigger_seconds_from_timestamps(
    total_duration: float,
    keyword_content: str = "",
    asr_tokens: Optional[List[Any]] = None,
    asr_timestamps: Optional[List[Any]] = None,
) -> Optional[float]:
    keyword = _normalize_alignment_text(keyword_content)
    if not keyword or not asr_tokens or not asr_timestamps:
        return None

    token_count = min(len(asr_tokens), len(asr_timestamps))
    if token_count <= 0:
        return None

    normalized_parts: List[str] = []
    token_indexes: List[int] = []
    for index in range(token_count):
        token_text = _normalize_alignment_text(asr_tokens[index])
        if not token_text:
            continue
        normalized_parts.append(token_text)
        token_indexes.extend([index] * len(token_text))

    normalized_text = "".join(normalized_parts)
    if not normalized_text:
        return None

    keyword_index = normalized_text.find(keyword)
    if keyword_index < 0:
        return None

    start_char_index = keyword_index
    end_char_index = keyword_index + len(keyword) - 1
    if end_char_index >= len(token_indexes):
        return None

    start_token_index = token_indexes[start_char_index]
    end_token_index = token_indexes[end_char_index]

    start_seconds = _safe_timestamp(asr_timestamps[start_token_index])
    end_seconds = _safe_timestamp(asr_timestamps[end_token_index])
    if start_seconds is None or end_seconds is None:
        return None

    if end_seconds < start_seconds:
        start_seconds, end_seconds = end_seconds, start_seconds

    trigger_seconds = (start_seconds + end_seconds) / 2
    return max(0.0, min(total_duration, trigger_seconds))


def _normalize_alignment_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"<\|[^>]+\|>", "", text)
    text = re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》【】\-_/\\|]+", "", text)
    return text


def _safe_timestamp(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_audio_duration(audio_bytes: bytes) -> float:
    """
    获取音频时长（秒）

    Args:
        audio_bytes: WAV音频字节流

    Returns:
        音频时长（秒），解析失败返回0.0
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0
