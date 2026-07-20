"""Fork-specific gateway regressions for Kamell's patched-main."""

from unittest.mock import patch

import gateway.run as gateway_run


def _runner() -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = None
    runner.session_store = None
    runner._session_model_overrides = {}
    runner._session_model_locks = {}
    return runner


def test_running_session_keeps_first_config_model_until_conversation_boundary() -> None:
    """A global config change must not hijack an already-running gateway chat."""
    runner = _runner()
    runtime = {
        "provider": "openrouter",
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

    with patch.object(
        gateway_run,
        "_resolve_gateway_model",
        side_effect=["first/model", "new-global/model"],
    ), patch.object(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        return_value=runtime,
    ):
        first_model, _ = runner._resolve_session_agent_runtime(session_key="session-1")
        second_model, _ = runner._resolve_session_agent_runtime(session_key="session-1")

    assert first_model == "first/model"
    assert second_model == "first/model"
    assert runner._session_model_locks == {"session-1": "first/model"}
