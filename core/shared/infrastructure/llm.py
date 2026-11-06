"""
大模型客户端 - 兼容 DeepSeek / 通义千问 (OpenAI 兼容 API)
"""
import hashlib
import json
from typing import Optional
import httpx
from openai import OpenAI
from core.shared import config


class LLMClient:
    """统一的大模型客户端"""

    def __init__(self, provider: Optional[str] = None, timeout_seconds: float = 600.0):
        self.provider = provider or config.LLM_PROVIDER

        timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        if self.provider == "deepseek":
            self._client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
                timeout=timeout,
            )
            self._model = config.DEEPSEEK_MODEL
        elif self.provider == "qwen":
            self._client = OpenAI(
                api_key=config.QWEN_API_KEY,
                base_url=config.QWEN_BASE_URL,
                timeout=timeout,
            )
            self._model = config.QWEN_MODEL
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """发送对话请求"""
        return self.chat_detailed(messages, temperature, max_tokens)["content"]

    def chat_detailed(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        usage = getattr(response, "usage", None)
        return {
            "content": response.choices[0].message.content or "",
            "provider": self.provider,
            "model": getattr(response, "model", None) or self._model,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        }

    def chat_with_cache(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        cache_manager=None,
    ) -> str:
        """带缓存的对话，相同参数短时间内不重复调用"""
        if cache_manager is None:
            return self.chat(messages, temperature, max_tokens)

        key_data = json.dumps(
            {
                "provider": self.provider,
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
        )
        cache_key = hashlib.md5(key_data.encode()).hexdigest()

        cached = cache_manager.get(cache_key)
        if cached:
            return cached

        result = self.chat(messages, temperature, max_tokens)
        cache_manager.set(cache_key, result, expire=config.CACHE_EXPIRE_LLM)
        return result

    def chat_stream(self, messages, temperature=0.7, max_tokens=2048):
        """流式对话，逐个 token 返回"""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @property
    def model_name(self) -> str:
        return self._model
