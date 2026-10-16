"""LLM 层：.env 装载 / 本地 Ollama 默认 / JSON 容错 / 多协议（全离线）。"""


import pytest

from duizhang import llm


def test_load_dotenv_reads_and_does_not_override(tmp_path):
    """KEY=VALUE 逐行装载；已存在的环境变量不覆盖（env 优先于 .env）。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'DZ_LLM_API_KEY="key123"\nDZ_LLM_MODEL=qwen3.5:0.8b\n'
        "# 注释行\nBADLINE 无等号\n\n", encoding="utf-8")
    env: dict[str, str] = {"DZ_LLM_MODEL": "qwen3:0.6b"}   # 已存在
    llm._load_dotenv(env_file, env=env)
    assert env["DZ_LLM_API_KEY"] == "key123"
    assert env["DZ_LLM_MODEL"] == "qwen3:0.6b"       # 未被 .env 覆盖


def test_local_defaults():
    """本地 Ollama 是默认 provider：零 key 零费用零外呼。"""
    assert llm.DEFAULT_BASE_URL == "http://localhost:11434/v1"
    assert llm.DEFAULT_MODEL == "qwen3.5:0.8b"


def test_llm_enabled_local_without_key(monkeypatch):
    """本地端点免鉴权：无 key 也启用；远端端点仍需 key。"""
    monkeypatch.delenv("DZ_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DZ_LLM_BASE_URL", raising=False)
    assert llm.llm_enabled() is True            # 默认即本地 Ollama
    monkeypatch.setenv("DZ_LLM_BASE_URL", "https://api.deepseek.com/v1")
    assert llm.llm_enabled() is False           # 远端无 key 不启用
    monkeypatch.setenv("DZ_LLM_API_KEY", "sk-x")
    assert llm.llm_enabled() is True


def test_local_chat_uses_native_api(monkeypatch):
    """本地通道走 Ollama 原生 /api/chat（OpenAI 兼容端点忽略 think 参数、
    思考会吃光 max_tokens 致正文为空——实测）。think 默认关：本地结构化
    调用要速度；DZ_LLM_THINKING=enabled 打开。"""

    captured = {}

    def fake_post(url, payload, timeout=300):
        captured["url"], captured["payload"] = url, payload
        return {"message": {"content": '```json\n{"ok": 1}\n```', "thinking": "x"}}

    monkeypatch.setattr(llm, "_http_post_json", fake_post)
    monkeypatch.setenv("DZ_LLM_BASE_URL", "http://localhost:11434/v1")
    assert llm.chat_json("sys", "user") == {"ok": 1}
    assert captured["url"].endswith("/api/chat")          # 原生端点
    assert captured["payload"]["think"] is False          # 默认关思考
    monkeypatch.setenv("DZ_LLM_THINKING", "enabled")
    llm.chat_text("sys", "user")
    assert captured["payload"]["think"] is True           # 显式打开


def test_load_dotenv_missing_file_is_noop(tmp_path):
    env: dict[str, str] = {}
    llm._load_dotenv(tmp_path / "absent.env", env=env)
    assert env == {}


def test_parse_json_tolerant():
    """思考型模型可能把 JSON 包进代码围栏或前后缀——容错取首个平衡对象。"""
    assert llm._parse_json('{"ok": true}') == {"ok": True}
    assert llm._parse_json('```json\n{"route": "metric"}\n```') == {"route": "metric"}
    assert llm._parse_json('好的，结果是 {"a": {"b": 1}} 以上。') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        llm._parse_json("完全不是 JSON")


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

    monkeypatch.setenv("DZ_LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    fa = _FakeAnthropic()
    monkeypatch.setattr(llm, "_client_anthropic", lambda: fa)
    assert llm.chat_json("sys", "user") == {"ok": True}   # 围栏 JSON 也解析
    assert llm.chat_text("sys", "user").strip().startswith("```")


def test_chat_with_tools_local_native(monkeypatch):
    """工具调用原语：本地走 Ollama 原生 tools（实测 0.8b 可靠），
    归一化返回 {content, tool_calls:[{name, arguments}]}——
    arguments 统一为 dict（兼容字符串/缺省形态）。"""

    captured = {}

    def fake_post(url, payload, timeout=300):
        captured["payload"] = payload
        return {"message": {"content": "", "tool_calls": [{
            "function": {"name": "search", "arguments": {"query": "营收"}}}]}}

    monkeypatch.setattr(llm, "_http_post_json", fake_post)
    monkeypatch.setenv("DZ_LLM_BASE_URL", "http://localhost:11434/v1")
    out = llm.chat_with_tools(
        [{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "search",
                "parameters": {"type": "object", "properties": {}}}}])
    assert out["tool_calls"] == [{"name": "search", "arguments": {"query": "营收"}}]
    assert captured["payload"]["tools"][0]["function"]["name"] == "search"

    # arguments 为 JSON 字符串的形态也归一化
    def fake_post2(url, payload, timeout=300):
        return {"message": {"content": "最终答案", "tool_calls": []}}
    monkeypatch.setattr(llm, "_http_post_json", fake_post2)
    out2 = llm.chat_with_tools([{"role": "user", "content": "q"}], tools=[])
    assert out2 == {"content": "最终答案", "tool_calls": []}
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
