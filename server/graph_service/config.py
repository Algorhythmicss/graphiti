from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None)
    falkordb_host: str | None = Field(None)
    falkordb_port: int | None = Field(None)
    falkordb_database: str | None = Field(None)
    db_backend: str = Field('neo4j')

    # Ambient proactive-memory tuning (deployment defaults; per-request overridable).
    ambient_default_token_budget: int = Field(512)
    ambient_default_draw_limit: int = Field(30)
    ambient_uncertainty_threshold: float = Field(0.6)
    ambient_decay_per_day: float = Field(0.005)
    # Two-level salience floors (cosine of fact vs transcript window). Calibrated
    # on the demo corpus with text-embedding-3-small; retune per deployment +
    # embedding model. min_top_relevance decides whether to speak at all;
    # min_relevance shapes which facts appear once speaking.
    ambient_min_top_relevance: float = Field(0.40)
    ambient_min_relevance: float = Field(0.22)

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
