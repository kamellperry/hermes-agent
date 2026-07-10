"""Regression tests for gateway model-picker capability detection."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _PluginAdapterProxy:
    """Model a plugin wrapper that delegates capabilities via __getattr__."""

    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)


def _make_event():
    return MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_bare_model_uses_picker_capability_delegated_by_plugin_proxy(
    tmp_path, monkeypatch
):
    """A plugin proxy can expose send_model_picker only on its instance.

    The old concrete-class lookup missed this capability and returned the
    plain-text `/model <name>` help instead of sending Telegram's picker.
    """
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-test\n  provider: openai-codex\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **kwargs: [
            {
                "slug": "openai-codex",
                "name": "OpenAI Codex",
                "models": ["gpt-test"],
                "total_models": 1,
                "is_current": True,
            }
        ],
    )

    target = SimpleNamespace(send_model_picker=AsyncMock(return_value=SendResult(success=True)))
    proxy = _PluginAdapterProxy(target)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: cast(Any, proxy)}
    runner._session_model_overrides = {}
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda source: "telegram:12345"
    runner._thread_metadata_for_source = lambda source, reply_to_message_id=None: {}
    runner._reply_anchor_for_event = lambda event: None

    result = await runner._handle_model_command(_make_event())

    assert result is None
    target.send_model_picker.assert_awaited_once()
