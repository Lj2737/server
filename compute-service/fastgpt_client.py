"""FastGPT chat workflow client for AI dialog."""
import asyncio
import json
from collections import deque
from typing import AsyncGenerator, Deque, Dict, List, Optional

import httpx
from loguru import logger

from config import (
    FASTGPT_API_KEY,
    FASTGPT_API_TIMEOUT,
    FASTGPT_API_URL,
    FASTGPT_DATASET_VARIABLE_KEY,
    FASTGPT_DATASET_VARIABLE_NAME,
    FASTGPT_MEMORY_TURNS,
)


class FastGPTClient:
    """Small async client for FastGPT chat completions."""

    def __init__(self) -> None:
        self._memories: Dict[str, Deque[Dict[str, str]]] = {}
        self._memory_locks: Dict[str, asyncio.Lock] = {}

    @staticmethod
    def _mask_secret(secret: str) -> str:
        if not secret:
            return "empty"
        if len(secret) <= 10:
            return f"len={len(secret)}"
        return f"{secret[:6]}...{secret[-4:]}(len={len(secret)})"

    async def chat(
        self,
        chat_id: str,
        user_content: str,
        knowledge_base_id: Optional[str] = None,
        memory_key: Optional[str] = None,
    ) -> Dict[str, str]:
        if not FASTGPT_API_KEY or FASTGPT_API_KEY.startswith("replace-with-"):
            raise RuntimeError("FASTGPT_API_KEY is not configured in compute-service/.env")
        if not user_content or not user_content.strip():
            raise RuntimeError("FastGPT user content is empty")

        payload = self._build_payload(
            chat_id=chat_id,
            user_content=user_content,
            knowledge_base_id=knowledge_base_id,
            stream=False,
            memory_messages=await self._get_memory_messages(memory_key),
        )

        headers = self._build_headers()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(FASTGPT_API_TIMEOUT),
            headers=headers,
        ) as client:
            response = await client.post(FASTGPT_API_URL, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if response.status_code == 401:
                    raise RuntimeError(
                        "FastGPT API authentication failed (401); "
                        f"key={self._mask_secret(FASTGPT_API_KEY)}, url={FASTGPT_API_URL}"
                    ) from e
                raise

        result = response.json()
        fastgpt_id = result.get("id") or chat_id
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(f"FastGPT response has no choices: {str(result)[:500]}")

        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if not content or not str(content).strip():
            raise RuntimeError(f"FastGPT response content is empty: {str(result)[:500]}")

        logger.info(
            f"FastGPT chat completed | chatId={chat_id} | "
            f"fastgptId={fastgpt_id} | "
            f"{FASTGPT_DATASET_VARIABLE_NAME}.{FASTGPT_DATASET_VARIABLE_KEY}={knowledge_base_id} | "
            f"memoryKey={memory_key or ''} | contentLength={len(content)}"
        )
        await self._append_memory(memory_key, user_content, str(content).strip())
        return {
            "id": str(fastgpt_id),
            "content": str(content).strip(),
        }

    async def stream_chat(
        self,
        chat_id: str,
        user_content: str,
        knowledge_base_id: Optional[str] = None,
        memory_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream FastGPT response chunks as dictionaries with id/content."""
        payload = self._build_payload(
            chat_id=chat_id,
            user_content=user_content,
            knowledge_base_id=knowledge_base_id,
            stream=True,
            memory_messages=await self._get_memory_messages(memory_key),
        )

        fastgpt_id = chat_id
        content_length = 0
        reply_content_parts: List[str] = []
        headers = self._build_headers()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(FASTGPT_API_TIMEOUT),
            headers=headers,
        ) as client:
            async with client.stream("POST", FASTGPT_API_URL, json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if response.status_code == 401:
                        raise RuntimeError(
                            "FastGPT API authentication failed (401); "
                            f"key={self._mask_secret(FASTGPT_API_KEY)}, url={FASTGPT_API_URL}"
                        ) from e
                    raise

                async for line in response.aiter_lines():
                    parsed = self._parse_stream_line(line)
                    if parsed is None:
                        continue
                    if parsed == "[DONE]":
                        break

                    chunk_id = parsed.get("id")
                    if chunk_id:
                        fastgpt_id = str(chunk_id)

                    content = self._extract_delta_content(parsed)
                    if content:
                        content_length += len(content)
                        reply_content_parts.append(content)
                        yield {
                            "id": fastgpt_id,
                            "content": content,
                        }

        reply_content = "".join(reply_content_parts).strip()
        if reply_content:
            await self._append_memory(memory_key, user_content, reply_content)
        logger.info(
            f"FastGPT stream completed | chatId={chat_id} | fastgptId={fastgpt_id} | "
            f"{FASTGPT_DATASET_VARIABLE_NAME}.{FASTGPT_DATASET_VARIABLE_KEY}={knowledge_base_id} | "
            f"memoryKey={memory_key or ''} | contentLength={content_length}"
        )

    def _build_payload(
        self,
        chat_id: str,
        user_content: str,
        knowledge_base_id: Optional[str],
        stream: bool,
        memory_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        if not FASTGPT_API_KEY or FASTGPT_API_KEY.startswith("replace-with-"):
            raise RuntimeError("FASTGPT_API_KEY is not configured in compute-service/.env")
        if not user_content or not user_content.strip():
            raise RuntimeError("FastGPT user content is empty")
        if not knowledge_base_id or not knowledge_base_id.strip():
            raise RuntimeError("FastGPT knowledge base id is required")

        messages: List[Dict[str, str]] = []
        if memory_messages:
            messages.extend(memory_messages)
        messages.append(
            {
                "role": "user",
                "content": user_content.strip(),
            }
        )

        return {
            "chatId": chat_id,
            "stream": stream,
            "detail": False,
            "variables": {
                FASTGPT_DATASET_VARIABLE_NAME: {
                    FASTGPT_DATASET_VARIABLE_KEY: knowledge_base_id.strip(),
                }
            },
            "messages": messages,
        }

    async def _get_memory_messages(
        self,
        memory_key: Optional[str],
    ) -> List[Dict[str, str]]:
        if not memory_key or FASTGPT_MEMORY_TURNS <= 0:
            return []
        lock = self._get_memory_lock(memory_key)
        async with lock:
            return [dict(message) for message in self._memories.get(memory_key, ())]

    async def _append_memory(
        self,
        memory_key: Optional[str],
        user_content: str,
        assistant_content: str,
    ) -> None:
        if not memory_key or FASTGPT_MEMORY_TURNS <= 0:
            return
        user_text = user_content.strip()
        assistant_text = assistant_content.strip()
        if not user_text or not assistant_text:
            return

        lock = self._get_memory_lock(memory_key)
        async with lock:
            max_messages = max(1, FASTGPT_MEMORY_TURNS) * 2
            memory = self._memories.get(memory_key)
            if memory is None:
                memory = deque(maxlen=max_messages)
                self._memories[memory_key] = memory
            elif memory.maxlen != max_messages:
                memory = deque(memory, maxlen=max_messages)
                self._memories[memory_key] = memory

            memory.append({"role": "user", "content": user_text})
            memory.append({"role": "assistant", "content": assistant_text})

    def _get_memory_lock(self, memory_key: str) -> asyncio.Lock:
        lock = self._memory_locks.get(memory_key)
        if lock is None:
            lock = asyncio.Lock()
            self._memory_locks[memory_key] = lock
        return lock

    @staticmethod
    def _build_headers() -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {FASTGPT_API_KEY}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_stream_line(line: str) -> Optional[dict | str]:
        if not line:
            return None
        text = line.strip()
        if not text:
            return None
        if text.startswith("data:"):
            text = text[5:].strip()
        if not text:
            return None
        if text == "[DONE]":
            return "[DONE]"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"FastGPT stream line ignored | line={text[:200]}")
            return None

    @staticmethod
    def _extract_delta_content(payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return ""

        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if content:
                return str(content)

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content:
                return str(content)

        content = choice.get("content")
        return str(content) if content else ""


fastgpt_client = FastGPTClient()
