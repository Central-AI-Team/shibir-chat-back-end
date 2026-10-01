"""One versioned, lossless resource snapshot for live replies, history and replay."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Citation(BaseModel):
    # Providers can attach page, section, URL, record IDs and provenance. Keep them
    # when serializing history rather than silently discarding visible metadata.
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    book: str
    chapter: str
    source_db: str
    content: str
    similarity: float | None = None
    rerank_score: float | None = None


class MessageResources(BaseModel):
    model_config = ConfigDict(extra='allow')
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    version: Literal[1] = 1
    sources: list[Citation] = Field(default_factory=list)
    web_results: list[dict[str, JsonValue]] = Field(default_factory=list)
    verification: dict[str, JsonValue] | None = None


def resource_snapshot(value=None, *, sources=None, web_results=None, verification=None):
    """Normalize server-generated resources; upgrade legacy sources-only rows on read."""
    if isinstance(value, MessageResources):
        data = value.model_dump(mode='json')
    else:
        data = dict(value or {})
    if 'sources' not in data and sources is not None:
        data['sources'] = sources
    if 'web_results' not in data and web_results is not None:
        data['web_results'] = web_results
    if 'verification' not in data and verification is not None:
        data['verification'] = verification
    return MessageResources.model_validate(data)


class ResourceFields(BaseModel):
    """The envelope is canonical; top-level fields remain compatible with old clients."""
    resources: MessageResources = Field(default_factory=MessageResources)
    sources: list[Citation] = Field(default_factory=list)
    web_results: list[dict[str, JsonValue]] = Field(default_factory=list)
    verification: dict[str, JsonValue] | None = None

    @model_validator(mode='after')
    def resource_aliases(self):
        if 'resources' not in self.model_fields_set:
            self.resources = resource_snapshot(sources=self.sources, web_results=self.web_results,
                                               verification=self.verification)
        self.sources = self.resources.sources
        self.web_results = self.resources.web_results
        self.verification = self.resources.verification
        return self
