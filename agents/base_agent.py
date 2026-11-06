"""
Agent 基类 - 定义分析师角色的通用接口
"""
from typing import Optional
from core.shared.infrastructure.llm import LLMClient
from core.shared.infrastructure.cache import CacheManager


class BaseAgent:
    """分析师 Agent 基类"""

    def __init__(
        self,
        name: str,
        role_prompt: str,
        llm: Optional[LLMClient] = None,
        cache: Optional[CacheManager] = None,
    ):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm or LLMClient()
        self.cache = cache

    def analyze(self, context: str, temperature: float = 0.7) -> str:
        """执行一次分析"""
        messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": context},
        ]
        return self.llm.chat_with_cache(
            messages=messages,
            temperature=temperature,
            cache_manager=self.cache,
        )

    def analyze_raw(self, messages: list, temperature: float = 0.7) -> str:
        """自定义消息序列的分析（用于多轮/多人讨论）"""
        return self.llm.chat_with_cache(
            messages=messages,
            temperature=temperature,
            cache_manager=self.cache,
        )
