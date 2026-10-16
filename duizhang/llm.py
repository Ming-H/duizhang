"""LLM 层：默认智谱 GLM（OpenAI 兼容端点），任意兼容端点可换。

v0.5 起 LLM 是系统的大脑（查询理解/评分/改写/生成四个决策点）；
heuristic 规则模式保留为零 key 兜底与回归基线。

密钥安全：API key 只放 .env（gitignored），由 _load_dotenv 装载；
已存在的环境变量优先，绝不覆盖。密钥不进代码、不进文档、不进 git。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4.5-flash"   # 免费档；充值后 .env 改 glm-4.6 提质


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


def llm_enabled() -> bool:
    return bool(os.environ.get("DZ_LLM_API_KEY"))


def _base_url() -> str:
    return os.environ.get("DZ_LLM_BASE_URL", DEFAULT_BASE_URL)


def _use_anthropic() -> bool:
    """GLM Coding Plan 的配额只在 Anthropic 兼容端点（…/api/anthropic）
    有效——v4 按量端点会报余额不足（实测）。base_url 指向该端点时走
    Anthropic Messages 协议（Claude Code 同款通道）。"""
    return "/api/anthropic" in _base_url()


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI  # 延迟导入：heuristic 模式不要求安装 openai
    return OpenAI(
        api_key=os.environ["DZ_LLM_API_KEY"],
        base_url=_base_url(),
    )


@lru_cache(maxsize=1)
def _client_anthropic():
    from anthropic import Anthropic  # 延迟导入：仅 Coding Plan 端点需要
    return Anthropic(api_key=os.environ["DZ_LLM_API_KEY"], base_url=_base_url())


def _anthropic_text(system: str, user: str, max_tokens: int) -> str:
    """Anthropic Messages 协议单轮调用（glm-5.3 等常思模型走此通道）。
    注意：新版 SDK 无 temperature 参数；thinking 默认开启且计入
    max_tokens——512 的默认预算会被思考吃光导致正文为空（实测），
    此通道预算至少 ×8。"""
    r = _client_anthropic().messages.create(
        model=os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        max_tokens=_max_tokens(max_tokens * 8 if not os.environ.get("DZ_LLM_MAX_TOKENS") else max_tokens),
        system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def _max_tokens(default: int) -> int:
    """DZ_LLM_MAX_TOKENS 可覆盖：重思考模型（如 GLM）的 reasoning token
    计入 max_tokens，默认值会被思考吃光导致正文为空。"""
    return int(os.environ.get("DZ_LLM_MAX_TOKENS", default))


def _thinking_off(for_generation: bool = False) -> bool:
    """GLM 思考策略分层（仅智谱端点发送该扩展参数，deepseek 等兼容端点
    不认识，绝不附加）：

    - 结构化小任务（理解/路由/评分/改写）：默认关——省时省钱，枚举
      抽取不需要推理；
    - 生成（for_generation=True）：默认开——碎片化表格证据里的数字
      关联（区分主体区间与「同比增加」的变动幅度）需要推理，flash
      档关思考实测会把变动区间误当指标值；
    - DZ_LLM_THINKING=disabled 全关 / =enabled 全开。
    """
    base = os.environ.get("DZ_LLM_BASE_URL", DEFAULT_BASE_URL)
    if "bigmodel" not in base:
        return False
    pref = os.environ.get("DZ_LLM_THINKING", "auto")
    if pref == "disabled":
        return True
    if pref == "enabled":
        return False
    return not for_generation


def _chat_kwargs(max_tokens: int, for_generation: bool = False) -> dict:
    kw = {"max_tokens": _max_tokens(max_tokens)}
    if _thinking_off(for_generation):
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
    return kw


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


def chat_json(system: str, user: str, max_tokens: int = 512) -> dict:
    """要求 JSON 输出的单轮调用；解析失败抛 ValueError 由调用节点降级。"""
    if _use_anthropic():
        return _parse_json(_anthropic_text(system, user, max_tokens))
    resp = _client().chat.completions.create(
        model=os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
        **_chat_kwargs(max_tokens),
    )
    return _parse_json(resp.choices[0].message.content or "")


def chat_text(system: str, user: str, max_tokens: int = 800) -> str:
    if _use_anthropic():
        return _anthropic_text(system, user, max_tokens)
    resp = _client().chat.completions.create(
        model=os.environ.get("DZ_LLM_MODEL", DEFAULT_MODEL),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        **_chat_kwargs(max_tokens, for_generation=True),
    )
    return resp.choices[0].message.content or ""
