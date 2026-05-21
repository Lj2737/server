"""FastGPT chat workflow client for AI dialog."""
from typing import Dict, Optional

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
    """Small async client for FastGPT non-streaming chat completions."""

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

        payload = {
            "chatId": chat_id,
            "stream": False,
            "detail": False,
            "messages": [
                {
                    "role": "user",
                    "content": user_content.strip(),
                }
            ],
        }
        if not knowledge_base_id or not knowledge_base_id.strip():
            raise RuntimeError("FastGPT knowledge base id is required")

        payload["variables"] = {
            FASTGPT_DATASET_VARIABLE_NAME: {
                FASTGPT_DATASET_VARIABLE_KEY: knowledge_base_id.strip(),
            }
        }

        headers = {
            "Authorization": f"Bearer {FASTGPT_API_KEY}",
            "Content-Type": "application/json",
        }
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


fastgpt_client = FastGPTClient()
