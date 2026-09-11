from typing import Final

INSECURE_MASTER_KEYS: Final = frozenset({"sk-1234"})


def insecure_master_key_error(master_key: str | None) -> str | None:
    if master_key not in INSECURE_MASTER_KEYS:
        return None
    return (
        "LITELLM_MASTER_KEY is set to the example key 'sk-1234' from the docs. "
        "Publicly reachable gateways using this key have been compromised. "
        "Set a strong random master key (e.g. `python -c \"import secrets; print('sk-' + secrets.token_urlsafe(32))\"`)."
    )
