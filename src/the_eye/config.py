"""Centralized environment & OS Keyring settings for The-Eye."""

from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _lookup_os_keyring(provider: str, env_var: str) -> str:
    """Resolve an API key from OS Keyring (Windows Credential Manager / macOS Keychain)."""
    try:
        import keyring
    except Exception:
        return ""

    for user_key in (provider, env_var):
        try:
            val = keyring.get_password("the-eye", user_key)
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            continue

    get_cred = getattr(keyring, "get_credential", None)
    if callable(get_cred):
        for target in (f"the-eye:{provider}", f"key:{provider}", provider, env_var):
            try:
                cred = get_cred(target, None)
                pw = getattr(cred, "password", None) if cred is not None else None
                if pw and str(pw).strip():
                    return str(pw).strip()
            except Exception:
                continue
    return ""


class Settings(BaseSettings):
    """Runtime configuration loaded from `.env` / environment variables or OS Keyring."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vlm_model: str = Field(default="ollama:qwen3.5:9b", alias="THE_EYE_VLM_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        alias="OLLAMA_BASE_URL",
    )
    openrouter_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    gemini_api_key: SecretStr = Field(default=SecretStr(""), alias="GEMINI_API_KEY")
    embed_backend: str = Field(default="auto", alias="THE_EYE_EMBED_BACKEND")

    @model_validator(mode="after")
    def _fill_keys_from_keyring(self) -> Settings:
        if not self.openrouter_api_key.get_secret_value().strip():
            kr_or = _lookup_os_keyring("openrouter", "OPENROUTER_API_KEY")
            if kr_or:
                self.openrouter_api_key = SecretStr(kr_or)
        if not self.gemini_api_key.get_secret_value().strip():
            kr_gem = _lookup_os_keyring("gemini", "GEMINI_API_KEY")
            if kr_gem:
                self.gemini_api_key = SecretStr(kr_gem)
        return self
