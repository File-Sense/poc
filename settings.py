from os import path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        validate_default=False,
        extra="ignore",
    )
    ROOT_DIR: str = path.realpath(path.dirname(__file__))
    ALLOW_RESET: bool = True
