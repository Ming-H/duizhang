"""可选 LLM 层：OpenAI 兼容端点（DeepSeek / GLM / GPT 均可）。

不配 key 时项目完全可用——heuristic 模式是默认且一等公民；
配了 DZ_LLM_* 三个环境变量后，路由/评分/改写/生成四个决策点
换成模型调用，图结构不变。这是刻意的：模型每代都在变，
编排与检索的假设要能活过任何一版模型。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache


def llm_enabled() -> bool:
    return bool(os.environ.get("DZ_LLM_API_KEY"))


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI  # 延迟导入：heuristic 模式不要求安装 openai
    return OpenAI(
        api_key=os.environ["DZ_LLM_API_KEY"],
        base_url=os.environ.get("DZ_LLM_BASE_URL", "https://api.deepseek.com/v1"),
    )


def chat_json(system: str, user: str, max_tokens: int = 512) -> dict:
    """要求 JSON 输出的单轮调用；解析失败抛 ValueError 由调用节点降级。"""
    model = os.environ.get("DZ_LLM_MODEL", "deepseek-chat")
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    return json.loads(raw)


def chat_text(system: str, user: str, max_tokens: int = 800) -> str:
    model = os.environ.get("DZ_LLM_MODEL", "deepseek-chat")
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
