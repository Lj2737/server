"""FastGPT chat workflow client for AI dialog."""
import json
from typing import AsyncGenerator, Dict, Optional

import httpx
from loguru import logger

from config import (
    FASTGPT_API_KEY,
    FASTGPT_API_TIMEOUT,
    FASTGPT_API_URL,
    FASTGPT_DATASET_VARIABLE_KEY,
    FASTGPT_DATASET_VARIABLE_NAME,
)


class FastGPTClient:
    """Small async client for FastGPT chat completions."""

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
            f"contentLength={len(content)}"
        )
        return {
            "id": str(fastgpt_id),
            "content": str(content).strip(),
        }

    async def stream_chat(
        self,
        chat_id: str,
        user_content: str,
        knowledge_base_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream FastGPT response chunks as dictionaries with id/content."""
        payload = self._build_payload(
            chat_id=chat_id,
            user_content=user_content,
            knowledge_base_id=knowledge_base_id,
            stream=True,
        )

        fastgpt_id = chat_id
        content_length = 0
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
                        yield {
                            "id": fastgpt_id,
                            "content": content,
                        }

        logger.info(
            f"FastGPT stream completed | chatId={chat_id} | fastgptId={fastgpt_id} | "
            f"{FASTGPT_DATASET_VARIABLE_NAME}.{FASTGPT_DATASET_VARIABLE_KEY}={knowledge_base_id} | "
            f"contentLength={content_length}"
        )

    def _build_payload(
        self,
        chat_id: str,
        user_content: str,
        knowledge_base_id: Optional[str],
        stream: bool,
    ) -> Dict[str, object]:
        if not FASTGPT_API_KEY or FASTGPT_API_KEY.startswith("replace-with-"):
            raise RuntimeError("FASTGPT_API_KEY is not configured in compute-service/.env")
        if not user_content or not user_content.strip():
            raise RuntimeError("FastGPT user content is empty")
        if not knowledge_base_id or not knowledge_base_id.strip():
            raise RuntimeError("FastGPT knowledge base id is required")

        return {
            "chatId": chat_id,
            "stream": stream,
            "detail": False,
            "variables": {
                FASTGPT_DATASET_VARIABLE_NAME: {
                    FASTGPT_DATASET_VARIABLE_KEY: knowledge_base_id.strip(),
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": user_content.strip(),
                }
            ],
        }

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
