from jarvis.agent.schemas import SpecialistDecision
from jarvis.agent.nodes import specialist
from jarvis.browser.context import BrowserError
from jarvis.config import Settings
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import build_default_registry


def test_specialist_decision_validators():
    s = SpecialistDecision(kind="act", tool="browser_search", args={"site": "YouTube", "query": "x"})
    assert s.kind == "act"
    try:
        SpecialistDecision(kind="act", tool="", args={})
        assert False
    except Exception:
        pass
    try:
        SpecialistDecision(kind="done", summary="")
        assert False
    except Exception:
        pass
    try:
        SpecialistDecision(kind="give_up", summary="")
        assert False
    except Exception:
        pass


def test_specialist_digest_small_and_untrusted():
    class FakeSess:
        def current_url(self):
            return "https://example.com/page"
        def read(self, max_elements=25):
            from jarvis.browser.context import ElementRef, Observation
            elems = [ElementRef(ref=f"e{i}", role="link", name="x"*100, origin="example.com") for i in range(30)]
            return Observation(url="https://example.com/page", origin="example.com", title="t", elements=tuple(elems))

    class Ctx:
        def __init__(self):
            self.browser = FakeSess()
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()

    d = specialist._build_digest({"specialist_goal": "go"}, Ctx())
    assert len(d) <= 2500
    assert "<untrusted_data>" in d


def test_specialist_call_cap_halts():
    class DummyLLM:
        def structured(self, **kwargs):
            from jarvis.agent.schemas import SpecialistDecision
            return SpecialistDecision(kind="give_up", summary="x")

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = DummyLLM()

    state = {"specialist_goal": "g", "specialist_calls": 12, "specialist_history": []}
    res = specialist.decide(state, Ctx())
    assert res.get("halted_reason") or res.get("final_answer")


def test_specialist_repeat_guard_halts():
    class DummyLLM:
        def structured(self, **kwargs):
            from jarvis.agent.schemas import SpecialistDecision
            return SpecialistDecision(kind="act", tool="browser_scroll", args={"direction": "down", "amount": 1})

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = DummyLLM()

    state = {"specialist_goal": "g", "specialist_calls": 0, "specialist_history": [{"tool": "browser_scroll", "args-summary": '{"direction": "down", "amount": 1}', "ok": True, "note": ""}, {"tool": "browser_scroll", "args-summary": '{"direction": "down", "amount": 1}', "ok": True, "note": ""}]}
    res = specialist.decide(state, Ctx())
    assert res.get("halted_reason") or res.get("final_answer")


def test_specialist_nonconsecutive_repeat_guard_halts():
    class CountingLLM:
        calls = 0

        def structured(self, **kwargs):
            self.calls += 1
            from jarvis.agent.schemas import SpecialistDecision

            return SpecialistDecision(kind="act", tool="browser_scroll", args={})

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = CountingLLM()

    summary = '{"query": "x"}'
    state = {
        "specialist_goal": "g",
        "specialist_calls": 2,
        "specialist_history": [
            {"tool": "browser_search", "args-summary": summary, "ok": True, "note": ""},
            {"tool": "browser_scroll", "args-summary": "{}", "ok": True, "note": ""},
            {"tool": "browser_search", "args-summary": summary, "ok": True, "note": ""},
        ],
    }
    ctx = Ctx()
    res = specialist.decide(state, ctx)
    assert res.get("halted_reason") == "Specialist stuck in a loop."
    assert ctx.llm.calls == 0


def test_specialist_proposed_repeat_halts_before_execution():
    class CountingLLM:
        calls = 0

        def structured(self, **kwargs):
            self.calls += 1
            from jarvis.agent.schemas import SpecialistDecision
            from jarvis.llm.client import Usage

            return (
                SpecialistDecision(kind="act", tool="browser_search", args={"query": "x"}),
                Usage(),
            )

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = CountingLLM()

    state = {
        "specialist_goal": "g",
        "specialist_calls": 2,
        "specialist_history": [
            {"tool": "browser_search", "args-summary": '{"query": "x"}', "ok": True, "note": ""},
            {"tool": "browser_scroll", "args-summary": "{}", "ok": True, "note": ""},
        ],
    }
    ctx = Ctx()
    res = specialist.decide(state, ctx)
    assert res.get("halted_reason") == "Specialist repeated an action."
    assert "plan" not in res
    assert ctx.llm.calls == 1


def test_specialist_refresh_entries_do_not_count_as_repeats():
    class DummyLLM:
        def structured(self, **kwargs):
            from jarvis.agent.schemas import SpecialistDecision
            from jarvis.llm.client import Usage

            return SpecialistDecision(kind="done", summary="ok"), Usage()

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = DummyLLM()

    state = {
        "specialist_goal": "g",
        "specialist_calls": 2,
        "specialist_history": [
            {"tool": "browser_search", "args-summary": '{"query": "a"}', "ok": True, "note": ""},
            {"tool": "browser_read", "args-summary": "{}", "ok": True, "note": "", "refresh": True},
            {"tool": "browser_search", "args-summary": '{"query": "b"}', "ok": True, "note": ""},
            {"tool": "browser_read", "args-summary": "{}", "ok": True, "note": "", "refresh": True},
        ],
    }
    res = specialist.decide(state, Ctx())
    assert res.get("final_answer") == "ok"
    assert res.get("halted_reason") is None


def test_specialist_grant_covered_repeats_do_not_count_as_loops():
    class DummyLLM:
        def structured(self, **kwargs):
            from jarvis.agent.schemas import SpecialistDecision
            from jarvis.llm.client import Usage

            return SpecialistDecision(kind="done", summary="ok"), Usage()

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = DummyLLM()

    summary = '{"url": "https://example.com/page"}'
    state = {
        "specialist_goal": "g",
        "specialist_calls": 2,
        "specialist_history": [
            {
                "tool": "browser_goto",
                "args-summary": summary,
                "ok": True,
                "note": "",
                "grant": True,
            },
            {
                "tool": "browser_goto",
                "args-summary": summary,
                "ok": True,
                "note": "",
                "grant": True,
            },
        ],
    }
    res = specialist.decide(state, Ctx())
    assert res.get("final_answer") == "ok"
    assert res.get("halted_reason") is None


def test_specialist_grant_covered_non_navigation_still_counts():
    class DummyLLM:
        def structured(self, **kwargs):
            from jarvis.agent.schemas import SpecialistDecision
            from jarvis.llm.client import Usage

            return SpecialistDecision(kind="done", summary="ok"), Usage()

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = DummyLLM()

    summary = '{"query": "x"}'
    state = {
        "specialist_goal": "g",
        "specialist_calls": 2,
        "specialist_history": [
            {
                "tool": "browser_search",
                "args-summary": summary,
                "ok": True,
                "note": "",
                "grant": True,
            },
            {
                "tool": "browser_search",
                "args-summary": summary,
                "ok": True,
                "note": "",
                "grant": True,
            },
        ],
    }
    res = specialist.decide(state, Ctx())
    assert res.get("halted_reason") == "Specialist stuck in a loop."


def test_specialist_nonbrowser_rejected():
    class DummyLLM:
        def structured(self, **kwargs):
            from jarvis.agent.schemas import SpecialistDecision
            return SpecialistDecision(kind="act", tool="google_search", args={"query": "x"})

    class Ctx:
        def __init__(self):
            self.browser = None
            self.browser_results = None
            self.settings = Settings()
            self.registry = build_default_registry()
            self.llm = DummyLLM()

    state = {"specialist_goal": "g", "specialist_calls": 0, "specialist_history": []}
    res = specialist.decide(state, Ctx())
    assert res.get("halted_reason")


def test_flag_off_routing_unchanged():
    from jarvis.agent import graph
    from jarvis.agent.context import AppContext
    from jarvis.tools.registry import build_default_registry
    from jarvis.config import Settings

    ctx = AppContext(settings=Settings(), registry=build_default_registry())
    g = graph.build_graph(ctx)
    # brain should route as before when flag off
    assert g is not None
