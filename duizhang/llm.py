"""LLM 层：默认本地 Ollama（零 key 零费用零外呼），任意远端可换。

v0.5.1 起 LLM 默认走本地（qwen3.5:0.8b @ Ollama）——全本地 Agentic RAG；
接远端（任意 OpenAI 兼容端点 / Anthropic 兼容端点）只需 .env 覆盖。

通道选择（按 base_url 自动判定）：
- 本地端点（localhost/127.0.0.1）→ Ollama **原生 /api/chat**：
  OpenAI 兼容端点实测忽略 think 参数、思考吃光 max_tokens 致正文为空；
  原生端点 think 有效（结构化调用默认关，DZ_LLM_THINKING=enabled 打开）
- 含 /api/anthropic → Anthropic Messages 协议（常思模型注意思考计入
  max_tokens，此通道预算 ×8）
- 其余 → OpenAI 兼容 chat/completions

密钥：本地免鉴权；远端密钥只放 .env（gitignored），_load_dotenv 装载
且环境变量优先。密钥不进代码、不进文档、不进 git。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # 本地 Ollama
DEFAULT_MODEL = "qwen3.5:0.8b"


def _load_dotenv(path: str | Path = ".env",
                 env: MutableMapping[str, str] | None = None) -> None:
    """极简 .env 装载：KEY=VALUE 逐行（支持引号/注释）；env 优先于 .env。"""
    target = env if env is not None else os.environ
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in target:
            target[key] = val


_load_dotenv()   # 导入即装载（仓库根/当前目录的 .env，不存在则跳过）


def _base_url() -> str:
    return os.environ.get("DZ_LLM_BASE_URL", DEFAULT_BASE_URL)


def _use_local() -> bool:
    return "localhost" in _base_url() or "127.0.0.1" in _base_url()


def llm_enabled() -> bool:
    """本地端点免鉴权（无 key 即启用）；远端端点需要 key。"""
    if os.environ.get("DZ_LLM_API_KEY"):
        return True
    return _use_local()


# ── 本地通道：Ollama 原生 /api/chat ────────────────────


def _http_post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    """stdlib POST（不引第三方依赖；本地调用毫秒级建连）。"""
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _local_think() -> bool:
    """本地思考默认关：0.8b 开思考延迟约 ×5（14s vs 3s，实测）且吃光
    预算；结构化抽取靠规则兜底，生成弱由 verify 闸门降级抽取式——
    系统纪律本来就兜得住。DZ_LLM_THINKING=enabled 可打开试质量。"""
    return os.environ.get("DZ_LLM_THINKING", "disabled") == "enabled"


def _local_chat(system: str, user: str, max_tokens: int) -> str:
    base = _base_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    r = _http_post_json(f"{base}/api/chat", {
        "model": os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        "stream": False,
        "think": _local_think(),
        "options": {"temperature": 0, "num_predict": _max_tokens(max_tokens)},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]})
    return (r.get("message") or {}).get("content") or ""


# ── 远端通道 ────────────────────────────────────────────


def _use_anthropic() -> bool:
    """Anthropic 兼容端点（含 /api/anthropic）走 Messages 协议。"""
    return "/api/anthropic" in _base_url()


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI  # 延迟导入：本地/无 key 模式不要求安装
    # v0.6.8 延迟治理：远端尾延迟实测 20s+（zhipu 偶发慢响应），无超时
    # 会把单问拖到 30s+；SDK 层 60s 超时 + 1 次重试砍长尾
    return OpenAI(
        api_key=os.environ.get("DZ_LLM_API_KEY", "local"),
        base_url=_base_url(),
        timeout=60.0, max_retries=1,
    )


@lru_cache(maxsize=1)
def _client_anthropic():
    from anthropic import Anthropic  # 延迟导入：仅 Anthropic 端点需要
    return Anthropic(api_key=os.environ["DZ_LLM_API_KEY"], base_url=_base_url())


def _anthropic_text(system: str, user: str, max_tokens: int) -> str:
    """Anthropic Messages 协议单轮调用。注意：新版 SDK 无 temperature
    参数；常思模型思考计入 max_tokens——512 会被思考吃光导致正文为空
    （实测），此通道预算 ×8（DZ_LLM_MAX_TOKENS 显式设置时不放大）。"""
    r = _client_anthropic().messages.create(
        model=os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        max_tokens=_max_tokens(max_tokens * 8 if not os.environ.get("DZ_LLM_MAX_TOKENS") else max_tokens),
        system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def _max_tokens(default: int) -> int:
    """DZ_LLM_MAX_TOKENS 可覆盖输出预算。"""
    return int(os.environ.get("DZ_LLM_MAX_TOKENS", default))


def _parse_json(raw: str) -> dict:
    """容错解析：剥 ```json 围栏、取首个花括号平衡块。失败抛 ValueError
    由调用节点降级到规则。"""
    s = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", s, re.S)
    if fence:
        s = fence.group(1)
    start = s.find("{")
    if start < 0:
        raise ValueError("no json object in output")
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("unbalanced json object")


# ── 统一入口 ────────────────────────────────────────────


def chat_json(system: str, user: str, max_tokens: int = 512) -> dict:
    """要求 JSON 输出的单轮调用；解析失败抛 ValueError 由调用节点降级。"""
    if _use_local():
        return _parse_json(_local_chat(system, user, max_tokens))
    if _use_anthropic():
        return _parse_json(_anthropic_text(system, user, max_tokens))
    resp = _client().chat.completions.create(
        model=os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=_max_tokens(max_tokens),
    )
    return _parse_json(resp.choices[0].message.content or "")


def chat_text(system: str, user: str, max_tokens: int = 800) -> str:
    if _use_local():
        return _local_chat(system, user, max_tokens)
    if _use_anthropic():
        return _anthropic_text(system, user, max_tokens)
    resp = _client().chat.completions.create(
        model=os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=_max_tokens(max_tokens),
    )
    return resp.choices[0].message.content or ""


def chat_with_tools(messages: list[dict], tools: list[dict],
                    max_tokens: int = 1024) -> dict:
    """工具调用单步原语（本地 Ollama 原生 tools；qwen3.5:0.8b 实测可用）。
    返回归一化 {content: str, tool_calls: [{name, arguments: dict}]}——
    arguments 兼容字符串/缺省形态。远端模式不支持工具：调用方回退固定管线。"""
    if not _use_local():
        raise NotImplementedError("tools 仅本地通道支持（远端走固定管线）")
    base = _base_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    r = _http_post_json(f"{base}/api/chat", {
        "model": os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        "stream": False,
        "think": _local_think(),
        "tools": tools,
        "options": {"temperature": 0, "num_predict": _max_tokens(max_tokens)},
        "messages": messages})
    m = r.get("message") or {}
    calls = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append({"name": fn.get("name", ""), "arguments": args or {}})
    return {"content": m.get("content") or "", "tool_calls": calls}
