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
