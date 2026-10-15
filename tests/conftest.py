"""测试永不外呼 LLM：有 .env 的机器上 mode=auto 会被强制解析为
heuristic；需要 llm 行为的测试自行 monkeypatch llm_enabled（见
test_llm_fallback_visible_in_trace / test_understand_node_llm_and_fallback）。
保证套件密封、免费、可进 CI。"""

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.delenv("DZ_LLM_API_KEY", raising=False)
