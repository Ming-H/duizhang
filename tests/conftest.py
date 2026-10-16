"""测试永不外呼 LLM：base_url 固定为远端占位（无 key → llm_enabled
False → auto 解析 heuristic），本地 Ollama 也不会被测试触碰；需要 llm
行为的测试自行 monkeypatch llm_enabled（见 test_llm_fallback_visible_
in_trace / test_understand_node_llm_and_fallback）。保证套件密封、
免费、可进 CI、与机器上是否装 Ollama 无关。"""

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.delenv("DZ_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DZ_LLM_BASE_URL", "https://placeholder.invalid/v1")
