"""LLM 层：.env 装载 / 本地 Ollama 默认 / JSON 容错 / 多协议（全离线）。"""


import pytest

from tally import llm


def test_load_dotenv_reads_and_does_not_override(tmp_path):
    """KEY=VALUE 逐行装载；已存在的环境变量不覆盖（env 优先于 .env）。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'TALLY_LLM_API_KEY="key123"\nTALLY_LLM_MODEL=qwen3.5:0.8b\n'
        "# 注释行\nBADLINE 无等号\n\n", encoding="utf-8")
    env: dict[str, str] = {"TALLY_LLM_MODEL": "qwen3:0.6b"}   # 已存在
    llm._load_dotenv(env_file, env=env)
    assert env["TALLY_LLM_API_KEY"] == "key123"
    assert env["TALLY_LLM_MODEL"] == "qwen3:0.6b"       # 未被 .env 覆盖


def test_local_defaults():
    """本地 Ollama 是默认 provider：零 key 零费用零外呼。"""
    assert llm.DEFAULT_BASE_URL == "http://localhost:11434/v1"
    assert llm.DEFAULT_MODEL == "qwen3.5:0.8b"


def test_llm_enabled_local_without_key(monkeypatch):
    """本地端点免鉴权：无 key 也启用；远端端点仍需 key。"""
    monkeypatch.delenv("TALLY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TALLY_LLM_BASE_URL", raising=False)
    assert llm.llm_enabled() is True            # 默认即本地 Ollama
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://api.deepseek.com/v1")
    assert llm.llm_enabled() is False           # 远端无 key 不启用
    monkeypatch.setenv("TALLY_LLM_API_KEY", "sk-x")
    assert llm.llm_enabled() is True


def test_local_chat_uses_native_api(monkeypatch):
    """本地通道走 Ollama 原生 /api/chat（OpenAI 兼容端点忽略 think 参数、
    思考会吃光 max_tokens 致正文为空——实测）。think 默认关：本地结构化
    调用要速度；TALLY_LLM_THINKING=enabled 打开。"""

    captured = {}

    def fake_post(url, payload, timeout=300):
        captured["url"], captured["payload"] = url, payload
        return {"message": {"content": '```json\n{"ok": 1}\n```', "thinking": "x"}}

    monkeypatch.setattr(llm, "_http_post_json", fake_post)
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "http://localhost:11434/v1")
    assert llm.chat_json("sys", "user") == {"ok": 1}
    assert captured["url"].endswith("/api/chat")          # 原生端点
    assert captured["payload"]["think"] is False          # 默认关思考
    monkeypatch.setenv("TALLY_LLM_THINKING", "enabled")
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

    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
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
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "http://localhost:11434/v1")
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
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    assert llm._use_anthropic() is True
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    assert llm._use_anthropic() is False


def test_auto_mode_resolution(monkeypatch):
    """mode=auto：有 key 走 llm（LLM 是大脑），无 key 走 heuristic（兜底），
    trace 记录解析结果；显式模式原样通过。"""
    from tally.schemas import new_state
    monkeypatch.delenv("TALLY_LLM_API_KEY", raising=False)
    st = new_state("q", "auto")
    assert st["mode"] == "heuristic"
    assert any("mode:heuristic" in t for t in st["trace"])
    monkeypatch.setattr(llm, "llm_enabled", lambda: True)
    st2 = new_state("q", "auto")
    assert st2["mode"] == "llm"
    assert any("mode:llm" in t for t in st2["trace"])
    st3 = new_state("q", "heuristic")
    assert st3["mode"] == "heuristic" and st3["trace"] == ["start"]


class _FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _FakeTC:
    def __init__(self, name, arguments):
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResp:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, captured, resp):
        self.captured, self.resp = captured, resp

    def create(self, **kw):
        self.captured.update(kw)
        return self.resp


class _FakeOpenAI:
    def __init__(self, captured, resp):
        self.chat = type("C", (), {"completions": _FakeCompletions(captured, resp)})()


def test_chat_with_tools_remote_openai_compat(monkeypatch):
    """v0.7.1：远端 OpenAI 兼容通道接通 function calling——修前显式
    NotImplementedError，agent 环在远端档位永远走 fallback 固定管线
    （DESIGN-v0.7 §8 实测记录的通道能力差异，本测试即其收口）。
    归一化契约与本地通道一致：{content, tool_calls:[{name, arguments:dict}]}，
    arguments 兼容 SDK 的 JSON 字符串形态。"""
    captured: dict = {}
    fake = _FakeOpenAI(captured, _FakeResp(_FakeMessage("", [
        _FakeTC("search", '{"query": "营收", "company": "欧科亿"}')])))
    monkeypatch.setattr(llm, "_client", lambda: fake)
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
    monkeypatch.setenv("TALLY_LLM_API_KEY", "test-key")
    out = llm.chat_with_tools(
        [{"role": "user", "content": "欧科亿营收"}],
        tools=[{"type": "function", "function": {"name": "search",
                "parameters": {"type": "object", "properties": {}}}}])
    assert out == {"content": "",
                   "tool_calls": [{"name": "search",
                                   "arguments": {"query": "营收", "company": "欧科亿"}}]}
    # tools 原样透传（注册表格式与 OpenAI 协议同构）
    assert captured["tools"][0]["function"]["name"] == "search"


def test_chat_with_tools_remote_normalizes_messages(monkeypatch):
    """远端协议消息规范化：agent 环喂的是宽松格式（assistant.tool_calls
    只有 function{name,arguments}、tool 消息无 id）——OpenAI 协议要求
    assistant 的 tool_calls 项带 id+type、tool 消息带 tool_call_id。
    id 按会话内位置确定性生成（call_1..call_N），每个请求自洽即可。"""
    captured: dict = {}
    fake = _FakeOpenAI(captured, _FakeResp(_FakeMessage("证据足够", None)))
    monkeypatch.setattr(llm, "_client", lambda: fake)
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("TALLY_LLM_API_KEY", "test-key")
    msgs = [
        {"role": "user", "content": "对比欧科亿和海伦哲营收"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "营收", "company": "欧科亿"}}},
            {"function": {"name": "search", "arguments": {"query": "营收", "company": "海伦哲"}}}]},
        {"role": "tool", "content": "[{\"doc\": \"a\"}]"},
        {"role": "tool", "content": "[{\"doc\": \"b\"}]"},
    ]
    out = llm.chat_with_tools(msgs, tools=[])
    assert out == {"content": "证据足够", "tool_calls": []}
    sent = captured["messages"]
    a_tc = sent[1]["tool_calls"]
    assert all(tc.get("id") and tc.get("type") == "function" for tc in a_tc)
    assert [t.get("tool_call_id") for t in sent[2:4]] == [a_tc[0]["id"], a_tc[1]["id"]]
    # 原消息列表不被就地污染（宽松格式保留给本地通道复用）
    assert "id" not in msgs[1]["tool_calls"][0]


def test_chat_with_tools_anthropic_channel_still_unsupported(monkeypatch):
    """Anthropic 兼容通道（Messages 协议 tool_use 块）仍未接——显式失败
    优于半成品协议转换；调用方（agent 环）照旧 fallback 固定管线。"""
    monkeypatch.setenv("TALLY_LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
    monkeypatch.setenv("TALLY_LLM_API_KEY", "test-key")
    import pytest
    with pytest.raises(NotImplementedError):
        llm.chat_with_tools([{"role": "user", "content": "q"}], tools=[])
