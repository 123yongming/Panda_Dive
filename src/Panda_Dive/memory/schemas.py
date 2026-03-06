"""Schemas for Panda_Dive memory and context pipeline."""

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    """Candidate memory extracted from intermediate text."""

    fact_type: str = Field(description="Type of fact, e.g. preference or knowledge.")
    content: str = Field(description="Extracted memory content.")
    confidence: float = Field(ge=0.0, le=1.0)
    source_urls: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    source_run_id: str | None = None
    novelty: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = Field(default="active")


class MemoryFact(BaseModel):
    """Persistent memory fact."""

    id: str
    namespace: str
    fact_type: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    source_urls: list[str] = Field(default_factory=list)
    source_run_id: str
    source_message_ids: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    ttl_days: int | None = None
    status: str = Field(default="active")
    rank_score: float = Field(default=0.0)


class MemoryEpisode(BaseModel):
    """Persistent episodic memory for a finished research run."""

    id: str
    namespace: str
    topic: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    created_at: str
    rank_score: float = Field(default=0.0)


class MemoryQuery(BaseModel):
    """Query object for memory retrieval."""

    query: str
    task_context: str = ""
    topic: str | None = None
    top_k: int = Field(default=8, ge=1, le=100)


class MemoryBundle(BaseModel):
    """Retrieved memory bundle for prompt injection."""

    facts: list[MemoryFact] = Field(default_factory=list)
    episodes: list[MemoryEpisode] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
