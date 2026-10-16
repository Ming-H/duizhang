"""LLM 层：.env 装载 / 智谱默认值 / JSON 容错 / thinking 控制（全离线）。"""


import pytest

from duizhang import llm


def test_load_dotenv_reads_and_does_not_override(tmp_path):
    """KEY=VALUE 逐行装载；已存在的环境变量不覆盖（env 优先于 .env）。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'DZ_LLM_API_KEY="key123"\nDZ_LLM_MODEL=glm-4.6\n'
        "# 注释行\nBADLINE 无等号\n\n", encoding="utf-8")
    env: dict[str, str] = {"DZ_LLM_MODEL": "glm-4-flash"}   # 已存在
    llm._load_dotenv(env_file, env=env)
    assert env["DZ_LLM_API_KEY"] == "key123"
    assert env["DZ_LLM_MODEL"] == "glm-4-flash"      # 未被 .env 覆盖


def test_load_dotenv_missing_file_is_noop(tmp_path):
    env: dict[str, str] = {}
    llm._load_dotenv(tmp_path / "absent.env", env=env)
    assert env == {}


def test_zhipu_defaults():
    """智谱 GLM 是默认 provider：任意 OpenAI 兼容端点仍可用 env 覆盖。"""
    assert "open.bigmodel.cn/api/paas/v4" in llm.DEFAULT_BASE_URL
    assert llm.DEFAULT_MODEL == "glm-4.5-flash"   # 免费档，开箱可用


def test_parse_json_tolerant():
    """思考型模型可能把 JSON 包进代码围栏或前后缀——容错取首个平衡对象。"""
    assert llm._parse_json('{"ok": true}') == {"ok": True}
    assert llm._parse_json('```json\n{"route": "metric"}\n```') == {"route": "metric"}
    assert llm._parse_json('好的，结果是 {"a": {"b": 1}} 以上。') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        llm._parse_json("完全不是 JSON")


def test_chat_kwargs_thinking_gating(monkeypatch):
    """GLM 思考模式默认关（结构化小任务省时省钱）；仅智谱端点附加扩展参数，
    其它兼容端点（如 deepseek）不认识这个参数绝不发送。"""
    monkeypatch.delenv("DZ_LLM_THINKING", raising=False)
    monkeypatch.delenv("DZ_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DZ_LLM_MAX_TOKENS", raising=False)
    # 智谱端点（默认）：结构化小任务关思考，生成开思考
    assert llm._chat_kwargs(512)["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in llm._chat_kwargs(800, for_generation=True)
    # 显式全开思考 → 小任务也不附加
    monkeypatch.setenv("DZ_LLM_THINKING", "enabled")
    assert "extra_body" not in llm._chat_kwargs(512)
    # 显式全关 → 生成也附加 disabled 标记（明确关）
    monkeypatch.setenv("DZ_LLM_THINKING", "disabled")
    assert llm._chat_kwargs(800, for_generation=True)["extra_body"] == {"thinking": {"type": "disabled"}}
    # 非智谱端点 → 永不附加
    monkeypatch.delenv("DZ_LLM_THINKING", raising=False)
    monkeypatch.setenv("DZ_LLM_BASE_URL", "https://api.deepseek.com/v1")
    assert "extra_body" not in llm._chat_kwargs(512)


def test_anthropic_protocol_detection_and_dispatch(monkeypatch):
    """Coding Plan 配额只在 Anthropic 兼容端点有效（v4 按量端点报余额
    不足）——base_url 含 /api/anthropic 时走 Anthropic 协议，密钥/
    模型/JSON 容错语义不变。"""

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Resp:
        content = [_Block('```json\n{"ok": true}\n```')]

    class _FakeAnthropic:
        def __init__(self):
            self.calls = []

        class messages:
            @staticmethod
            def create(**kw):
                return _Resp()

    monkeypatch.delenv("DZ_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(llm, "_use_anthropic", lambda: True)
    fa = _FakeAnthropic()
    monkeypatch.setattr(llm, "_client_anthropic", lambda: fa)
    assert llm.chat_json("sys", "user") == {"ok": True}   # 围栏 JSON 也解析
    assert llm.chat_text("sys", "user").strip().startswith("```")


def test_base_url_detection(monkeypatch):
    monkeypatch.setenv("DZ_LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    assert llm._use_anthropic() is True
    monkeypatch.setenv("DZ_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    assert llm._use_anthropic() is False


def test_auto_mode_resolution(monkeypatch):
    """mode=auto：有 key 走 llm（LLM 是大脑），无 key 走 heuristic（兜底），
    trace 记录解析结果；显式模式原样通过。"""
    from duizhang.schemas import new_state
    monkeypatch.delenv("DZ_LLM_API_KEY", raising=False)
    st = new_state("q", "auto")
    assert st["mode"] == "heuristic"
    assert any("mode:heuristic" in t for t in st["trace"])
    monkeypatch.setattr(llm, "llm_enabled", lambda: True)
    st2 = new_state("q", "auto")
    assert st2["mode"] == "llm"
    assert any("mode:llm" in t for t in st2["trace"])
    st3 = new_state("q", "heuristic")
    assert st3["mode"] == "heuristic" and st3["trace"] == ["start"]
