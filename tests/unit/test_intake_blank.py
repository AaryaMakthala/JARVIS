from jarvis.agent.nodes.intake import intake


class DummyMem:
    def record_task(self, *args, **kwargs):
        return None


class DummyCtx:
    def __init__(self):
        self.memory = DummyMem()
        self.dry_run = False
        self.logger = __import__('logging').getLogger('test')


def test_blank_input_reaches_halt_branch():
    state = {"user_input": "   ", "source": "terminal"}
    ctx = DummyCtx()
    out = intake(state, ctx)
    assert out.get("halted_reason") == "No input received. Tell me what to do."
    assert out.get("plan") is None


def test_intake_resets_every_per_task_field_of_a_reused_state():
    stale = {
        "user_input": "   ",
        "source": "terminal",
        "final_answer": "stale answer",
        "spoken_answer": "stale speech",
        "screen_answer": "stale screen",
        "error": "stale error",
        "resolved_refs": {"it": "x"},
        "gated_resolved_paths": {"s1": ["C:/tmp/x"]},
        "gated_content_hashes": {"s1": ["abc"]},
        "gated_browser_readback_hashes": {"s1": "abc"},
        "pending_readback": {"step_id": "s1"},
        "specialist_goal": "old goal",
        "specialist_calls": 3,
        "specialist_history": [{"x": 1}],
        "specialist_phase": "observe",
        "memory_saved": True,
        "memory_saved_reason": "old",
        "memory_failure_logged": True,
    }
    out = intake(stale, DummyCtx())
    assert out["final_answer"] is None
    assert out["spoken_answer"] is None
    assert out["screen_answer"] is None
    assert out["error"] is None
    assert out["resolved_refs"] == {}
    assert out["gated_resolved_paths"] == {}
    assert out["gated_content_hashes"] == {}
    assert out["gated_browser_readback_hashes"] == {}
    assert out["pending_readback"] is None
    assert out["specialist_goal"] is None
    assert out["specialist_calls"] == 0
    assert out["specialist_history"] == []
    assert out["specialist_phase"] == "idle"
    assert out["memory_saved"] is False
    assert out["memory_saved_reason"] is None
    assert out["memory_failure_logged"] is False
