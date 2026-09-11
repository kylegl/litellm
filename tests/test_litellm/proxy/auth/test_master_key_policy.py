from typing import Final

from litellm.proxy.auth.master_key_policy import (
    INSECURE_MASTER_KEYS,
    insecure_master_key_error,
)

EXAMPLE_MASTER_KEY: Final = next(iter(INSECURE_MASTER_KEYS))


def test_insecure_master_key_error_refuses_example_key():
    error = insecure_master_key_error(master_key=EXAMPLE_MASTER_KEY)
    assert error is not None
    assert EXAMPLE_MASTER_KEY in error


def test_insecure_master_key_error_allows_strong_key():
    assert insecure_master_key_error(master_key="sk-strong-random-key") is None


def test_insecure_master_key_error_allows_missing_key():
    assert insecure_master_key_error(master_key=None) is None


def test_remediation_hint_survives_log_redaction():
    from litellm.litellm_core_utils.secret_redaction import redact_string

    error = insecure_master_key_error(master_key=EXAMPLE_MASTER_KEY)
    assert error is not None
    assert "secrets.token_urlsafe" in redact_string(error)
