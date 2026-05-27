"""
Disk caching utilities for LLM-generated content.

Avoids repeated API calls for:
  - Item intent labels (RecAug)
  - Hard negative candidates (SANS)
  - Item substitution candidates (RecAug)
  - Session boundaries (RecAug)
"""

import json
import os
import hashlib
from typing import Dict, Any, Optional
import pickle


class DiskCache:
    """Simple JSON-based disk cache with optional compression."""

    def __init__(self, cache_dir: str = "data/cache/"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str, suffix: str = "json") -> str:
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.{suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        path = self._path(key)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default
        return default

    def set(self, key: str, value: Any):
        path = self._path(key)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def get_or_set(self, key: str, factory):
        """Get from cache or compute and store."""
        result = self.get(key)
        if result is not None:
            return result
        result = factory()
        if result is not None:
            self.set(key, result)
        return result

    def load_json_dict(self, path: str) -> Dict[str, Any]:
        """Load a complete JSON dictionary cache file."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_json_dict(self, data: Dict[str, Any], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class LLMClient:
    """Wrapper for DeepSeek API via Anthropic-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-chat",
                 max_tokens: int = 512, temperature: float = 0.7,
                 request_interval: float = 1.0):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_interval = request_interval
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 system_prompt: str = "You are a helpful assistant that analyzes video game data.") -> str:
        """Generate text from LLM. Simple single-turn completion."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        import time
        time.sleep(self.request_interval)  # rate limiting after successful call
        return response.content[0].text
