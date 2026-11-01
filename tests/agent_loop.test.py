#!/usr/bin/env python3
"""Exercise the documented Agent Loop with deterministic models and fake tools.

Run from the repository root: python tests/agent_loop.test.py
Uses only the Python standard library; no API key or network is needed.
"""

from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def action(tool_name="lookup", **kwargs):
    return SimpleNamespace(
        type="tool_call", tool_name=tool_name, args=kwargs, content=None
    )


def final_answer(content="done"):
    return SimpleNamespace(type="final_answer", content=content)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        template = ROOT / "examples" / "minimal-agent-loop.md"
        source = re.search(
            r"```python\n(.*?)```", template.read_text(encoding="utf-8"), re.DOTALL
        )
        self.assertIsNotNone(source, "Agent Loop template needs a Python example")
        self.trace = Mock()
        self.ask_human = Mock(return_value=True)
        namespace = {
            "build_context": lambda **kwargs: deepcopy(kwargs["state"]),
            "trace": self.trace,
            "ask_human": self.ask_human,
            "summarize_state": lambda state: f"{len(state['steps'])} steps",
        }
        exec(compile(source.group(1), str(template), "exec"), namespace)
        self.run_agent = namespace["run_agent"]
        self.tool = SimpleNamespace(
            requires_confirmation=False,
            call=Mock(return_value={"ok": True, "data_summary": "found"}),
        )

    def model(self, *actions):
        pending = iter(actions)
        self.contexts = []

        def decide(context):
            self.contexts.append(context)
            return next(pending)

        return SimpleNamespace(decide=decide)

    def assert_feedback(self, error):
        steps = self.contexts[1]["steps"]
        self.assertEqual(len(steps), 1, "Next decision must see the failed step")
        self.assertEqual(steps[0]["observation"]["error"], error)
        self.assertFalse(steps[0]["observation"]["ok"])

    def test_unknown_tool_feedback_allows_recovery(self):
        model = self.model(action("missing"), action(query="retry"), final_answer())
        state = self.run_agent("find evidence", {"lookup": self.tool}, model)

        self.assert_feedback("unknown_tool")
        self.tool.call.assert_called_once_with(query="retry")
        self.assertEqual(len(state["steps"]), 2)
        self.assertTrue(state["steps"][1]["observation"]["ok"])
        self.assertEqual(state["final"], "done")
        self.assertEqual(self.trace.call_count, 3)

    def test_human_rejection_is_visible_without_executing_tool(self):
        self.tool.requires_confirmation = True
        self.ask_human.return_value = False
        model = self.model(action(), final_answer("cancelled"))
        state = self.run_agent("write file", {"lookup": self.tool}, model)

        self.assert_feedback("human_rejected")
        self.tool.call.assert_not_called()
        self.ask_human.assert_called_once()
        self.assertEqual(len(state["steps"]), 1)
        self.assertEqual(state["final"], "cancelled")

    def test_tool_errors_remain_visible_to_next_decision(self):
        for exc, error in (
            (TimeoutError(), "timeout"),
            (ValueError("invalid query"), "tool_exception"),
        ):
            with self.subTest(error=error):
                self.tool.call.side_effect = exc
                model = self.model(action(), final_answer())
                state = self.run_agent("search", {"lookup": self.tool}, model)
                self.assert_feedback(error)
                self.assertEqual(len(state["steps"]), 1)

    def test_step_limit_keeps_all_failed_attempts_for_diagnosis(self):
        model = self.model(action("missing"), action("missing"))
        state = self.run_agent("search", {}, model, max_steps=2)

        self.assertEqual(len(self.contexts), 2)
        self.assertEqual(len(state["steps"]), 2)
        self.assertEqual(state["final"]["reason"], "max_steps_exceeded")
        self.assertEqual(state["final"]["summary"], "2 steps")
        self.assertEqual(self.trace.call_count, 2)

    def test_successful_tool_result_is_visible_to_next_decision(self):
        model = self.model(action(query="agent"), final_answer("answer"))
        state = self.run_agent("search", {"lookup": self.tool}, model)

        self.assertEqual(
            self.contexts[1]["steps"][0]["observation"],
            {"ok": True, "data_summary": "found"},
        )
        self.assertEqual(state["final"], "answer")
        self.ask_human.assert_not_called()


if __name__ == "__main__":
    unittest.main()
