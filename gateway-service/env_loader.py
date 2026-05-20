"""
Lightweight .env loader.

Service-local .env values override existing OS environment variables by default.
"""
import os
from pathlib import Path


def load_env_file(filename: str = ".env", override: bool = True) -> None:
    """Load KEY=VALUE pairs from gateway-service/.env if it exists."""
    env_path = Path(__file__).resolve().parent / filename
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
