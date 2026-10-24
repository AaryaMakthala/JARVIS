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
