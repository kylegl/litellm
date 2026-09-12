import base64
import json
import os
import time
from pathlib import Path
from typing import Final
from unittest.mock import mock_open, patch

import httpx
import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError, RefreshAccessTokenError


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"


def test_auth_record_replacement_is_private_and_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    record: Final = {"access_token": "new-access", "refresh_token": "new-refresh", "expires_at": 4102444800}
    authenticator: Final = Authenticator()
    authenticator._write_auth_file({"access_token": "old-access"})
    authenticator._write_auth_file(record)
    assert json.loads((tmp_path / "auth.json").read_text()) == record
    assert Authenticator().get_access_token() == "new-access"
    if os.name == "posix":
        assert (tmp_path / "auth.json").stat().st_mode & 0o777 == 0o600
    assert sorted(path.name for path in tmp_path.iterdir()) == ["auth.json"]


def test_failed_auth_replacement_preserves_previous_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    previous: Final = {"access_token": "old-access", "refresh_token": "old-refresh"}
    (tmp_path / "auth.json").write_text(json.dumps(previous))
    with patch("litellm.llms.chatgpt.authenticator.os.replace", side_effect=OSError("private-path")):
        with pytest.raises(GetAccessTokenError, match="Could not persist ChatGPT credentials") as error:
            Authenticator()._write_auth_file({"access_token": "new-access"})
    assert "private-path" not in str(error.value)
    assert json.loads((tmp_path / "auth.json").read_text()) == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["auth.json"]


def test_noninteractive_missing_credentials_do_not_start_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_NONINTERACTIVE", "true")
    with patch("litellm.llms.chatgpt.authenticator._get_httpx_client") as client:
        with pytest.raises(GetAccessTokenError) as error:
            Authenticator().get_access_token()
    assert error.value.status_code == 401
    client.assert_not_called()


@pytest.mark.parametrize("missing_id_token", [False, True])
def test_refresh_rotation_persists_or_fails_closed_without_token_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, missing_id_token: bool
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_NONINTERACTIVE", "true")
    previous: Final = {"access_token": "old-access", "refresh_token": "old-refresh", "expires_at": 0}
    (tmp_path / "auth.json").write_text(json.dumps(previous))
    access: Final = _make_jwt({"exp": 4102444800})
    identity: Final = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "fictional-account"}})

    def refresh(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        assert json.loads(request.content)["refresh_token"] == "old-refresh"
        return httpx.Response(
            200,
            json={
                "access_token": access,
                "refresh_token": "rotated-secret",
                **({} if missing_id_token else {"id_token": identity}),
            },
        )

    with httpx.Client(transport=httpx.MockTransport(refresh)) as client:
        with patch("litellm.llms.chatgpt.authenticator._get_httpx_client", return_value=client):
            if missing_id_token:
                with pytest.raises(RefreshAccessTokenError) as error:
                    Authenticator()._refresh_tokens("old-refresh")
                assert "rotated-secret" not in str(error.value)
                assert access not in str(error.value)
                with pytest.raises(GetAccessTokenError) as noninteractive_error:
                    Authenticator().get_access_token()
                assert noninteractive_error.value.status_code == 401
                assert json.loads((tmp_path / "auth.json").read_text()) == previous
            else:
                assert Authenticator().get_access_token() == access
                assert json.loads((tmp_path / "auth.json").read_text())["refresh_token"] == "rotated-secret"
                assert Authenticator().get_access_token() == access
                assert Authenticator().get_account_id() == "fictional-account"
    assert "rotated-secret" not in caplog.text
    assert access not in caplog.text


def test_malformed_refresh_response_does_not_disclose_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=["private-token"]))
    ) as client:
        with patch("litellm.llms.chatgpt.authenticator._get_httpx_client", return_value=client):
            with pytest.raises(RefreshAccessTokenError) as error:
                Authenticator()._refresh_tokens("old-refresh")
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__ is True
