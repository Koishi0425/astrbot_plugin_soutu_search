"""「指令后抑制 LLM 联动」测试（工程师）。

背景：AstrBot 在命令 handler 返回结果后，若事件仍处于「唤醒命令」状态，会继续走
「默认 LLM 请求链路」，导致 LLM 人格插件（如 AstrNa）对「搜本/搜图」指令额外回一段人话
（例如对 NSFW 图说「不接 / 自己去识图网站」）。插件通过 ``should_call_llm(True)``
（禁默认 LLM）+ ``stop_event()``（止后续 handler）关闭该联动，默认开启、可配置关闭
（``suppress_llm_followup``）。

覆盖：
- ``suppress_llm_followup`` 默认 True（插件属性 + schema）
- 默认配置下 ``_suppress_followup`` 调用 ``should_call_llm(True)`` + ``stop_event()``
- ``suppress_llm_followup=False`` 时两者都不调用
- 假事件无这两个方法 / 方法抛异常 → 静默降级，不影响主流程
- 四个命令入口（book_cmd/sou_cmd/两个帮助类）确实在开头调用了抑制

复用 tests/test_core.py 的 astrbot 桩（import 即完成 sys.modules 装配）。
运行::
    python -m unittest tests.test_suppress_followup -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PLUGIN_ROOT.parent), str(PLUGIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tests.test_core as _stub  # noqa: E402,F401  (装配 astrbot 桩)

from astrbot_plugin_soutu_search.main import (  # noqa: E402
    ACCESS_DENIED_TEXT,
    SoutuSearchPlugin,
)


def run(coro):
    return asyncio.run(coro)


def collect(agen):
    return run(_collect(agen))


async def _collect(agen):
    return [x async for x in agen]


class RecordingEvent:
    """记录 ``should_call_llm`` / ``stop_event`` 调用的事件替身。"""

    def __init__(self, umo="umo-A", text=""):
        self.unified_msg_origin = umo
        self.message_str = text
        self.message_obj = type("M", (), {"message_id": "m1"})()
        self.should_call_llm_calls = []
        self.stop_event_calls = 0

    def get_message_str(self):
        return self.message_str

    def plain_result(self, text):
        return ("plain", text)

    def chain_result(self, comps):
        return ("chain", comps)

    def should_call_llm(self, call_llm):
        self.should_call_llm_calls.append(call_llm)

    def stop_event(self):
        self.stop_event_calls += 1


class BareEvent:
    """只有 plain_result 的极简事件（模拟老版本 AstrBot / 其它测试桩，无抑制 API）。"""

    def plain_result(self, text):
        return ("plain", text)


class ThrowingEvent(RecordingEvent):
    """should_call_llm 抛异常的事件，验证静默降级且不影响 stop_event。"""

    def should_call_llm(self, call_llm):
        raise RuntimeError("boom")


class TestSuppressFollowup(unittest.TestCase):
    def _plugin(self, **cfg):
        return SoutuSearchPlugin(object(), dict(cfg))

    def test_default_enabled(self):
        self.assertTrue(self._plugin().suppress_llm_followup)

    def test_explicitly_disabled(self):
        self.assertFalse(self._plugin(suppress_llm_followup=False).suppress_llm_followup)

    def test_suppress_calls_both_apis(self):
        ev = RecordingEvent()
        self._plugin()._suppress_followup(ev)
        self.assertEqual(ev.should_call_llm_calls, [True])
        self.assertEqual(ev.stop_event_calls, 1)

    def test_disabled_skips_both(self):
        ev = RecordingEvent()
        self._plugin(suppress_llm_followup=False)._suppress_followup(ev)
        self.assertEqual(ev.should_call_llm_calls, [])
        self.assertEqual(ev.stop_event_calls, 0)

    def test_missing_api_is_ignored(self):
        # 老版本 AstrBot / 其它测试桩没有这两个方法 → 静默降级，不抛异常
        self._plugin()._suppress_followup(BareEvent())

    def test_throwing_api_is_ignored(self):
        # 任一 API 抛异常都静默忽略，且不影响后续 API 的调用
        ev = ThrowingEvent()
        self._plugin()._suppress_followup(ev)
        self.assertEqual(ev.stop_event_calls, 1)

    def test_schema_default_true(self):
        schema = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertIs(schema["suppress_llm_followup"]["default"], True)


class TestHandlerCallsSuppress(unittest.TestCase):
    """验证四个命令入口确实在开头调用了抑制。

    用 whitelist 空白名单（fail-closed）让 handler 走「拒绝提示」分支，流程最短，
    同时确保抑制发生在访问控制判定之前（即只要指令命中就抑制，不取决于是否放行）。
    """

    def _denied_plugin(self):
        return SoutuSearchPlugin(object(), {"access_mode": "whitelist", "whitelist": []})

    def _assert_suppressed(self, agen, ev, text):
        out = collect(agen)
        self.assertEqual(ev.should_call_llm_calls, [True], text)
        self.assertEqual(ev.stop_event_calls, 1, text)
        self.assertEqual(out, [("plain", ACCESS_DENIED_TEXT)], text)

    def test_book_cmd_suppresses(self):
        p = self._denied_plugin()
        ev = RecordingEvent(text="/搜本")
        self._assert_suppressed(p.book_cmd(ev, ""), ev, "book_cmd")

    def test_book_help_cmd_suppresses(self):
        p = self._denied_plugin()
        ev = RecordingEvent(text="/搜本帮助")
        self._assert_suppressed(p.book_help_cmd(ev), ev, "book_help_cmd")

    def test_sou_cmd_suppresses(self):
        p = self._denied_plugin()
        ev = RecordingEvent(text="/搜图")
        self._assert_suppressed(p.sou_cmd(ev, ""), ev, "sou_cmd")

    def test_sou_help_cmd_suppresses(self):
        p = self._denied_plugin()
        ev = RecordingEvent(text="/搜图帮助")
        self._assert_suppressed(p.sou_help_cmd(ev), ev, "sou_help_cmd")


if __name__ == "__main__":
    unittest.main()
