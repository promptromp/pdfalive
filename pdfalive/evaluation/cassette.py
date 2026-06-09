"""VCR-style record/replay of structured LLM responses.

RecordingChatModel wraps a real chat model, persisting every structured
response to a JSON cassette file as it arrives. ReplayChatModel plays a
cassette back without any network calls, so evaluations of the deterministic
pipeline (feature extraction, corrections, postprocess fixups) are free and
reproducible.

Both classes duck-type the subset of the LangChain chat model interface that
the processors actually use: ``with_structured_output(schema)`` returning an
object with ``invoke(messages)``.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, Field


class CassetteMissError(Exception):
    """Raised when a replay request cannot be served from the cassette."""


class CassetteEntry(BaseModel):
    """One recorded LLM call: request fingerprint plus structured response."""

    request_hash: str = Field(description="SHA-256 fingerprint of the request messages and schema")
    schema_name: str = Field(description="Name of the structured output schema (e.g. 'TOC')")
    response: dict = Field(description="Structured response as a plain dict (model_dump)")
    usage_metadata: dict | None = Field(
        default=None, description="Actual token usage reported by the provider, when recorded with include_raw"
    )


class Cassette(BaseModel):
    """An ordered collection of recorded LLM calls, persisted as JSON."""

    entries: list[CassetteEntry] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Cassette":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def _hash_request(messages: list, schema_name: str) -> str:
    """Build a stable fingerprint for a structured LLM request.

    Uses each message's type and content plus the schema name. Identical
    prompts always produce identical hashes, so a replay mismatch means the
    pipeline now builds different prompts than when the cassette was recorded.
    """
    digest = hashlib.sha256()
    digest.update(schema_name.encode())
    for message in messages:
        digest.update(str(getattr(message, "type", message.__class__.__name__)).encode())
        digest.update(str(message.content).encode())
    return digest.hexdigest()


class _RecordingStructuredModel:
    """Wraps a structured model, persisting each response to the cassette."""

    def __init__(self, structured_model, schema: type[BaseModel], cassette: Cassette, cassette_path: Path) -> None:
        self._structured_model = structured_model
        self._schema = schema
        self._cassette = cassette
        self._cassette_path = cassette_path

    def invoke(self, messages: list) -> BaseModel | dict:
        response = self._structured_model.invoke(messages)

        # include_raw=True responses are dicts: {"raw": AIMessage, "parsed": model, "parsing_error": ...}.
        # Record the parsed model plus the provider's actual token usage.
        if isinstance(response, dict) and "parsed" in response:
            parsed = response["parsed"]
            raw = response.get("raw")
            usage_metadata = getattr(raw, "usage_metadata", None) if raw is not None else None
        else:
            parsed = response
            usage_metadata = None

        self._cassette.entries.append(
            CassetteEntry(
                request_hash=_hash_request(messages, self._schema.__name__),
                schema_name=self._schema.__name__,
                response=parsed.model_dump(),
                usage_metadata=usage_metadata,
            )
        )
        # Persist after every call so long multi-batch runs survive interruption.
        self._cassette.save(self._cassette_path)
        return response


class RecordingChatModel:
    """Chat-model wrapper that records structured responses to a cassette file."""

    def __init__(self, llm, cassette_path: Path) -> None:
        self._llm = llm
        self._cassette_path = Path(cassette_path)
        self._cassette = Cassette()

    def with_structured_output(self, schema: type[BaseModel], include_raw: bool = False) -> _RecordingStructuredModel:
        structured_model = (
            self._llm.with_structured_output(schema, include_raw=True)
            if include_raw
            else self._llm.with_structured_output(schema)
        )
        return _RecordingStructuredModel(
            structured_model=structured_model,
            schema=schema,
            cassette=self._cassette,
            cassette_path=self._cassette_path,
        )


class _ReplayStructuredModel:
    """Serves recorded responses in order, validated against the schema."""

    def __init__(self, replay_state: "ReplayChatModel", schema: type[BaseModel], include_raw: bool) -> None:
        self._replay_state = replay_state
        self._schema = schema
        self._include_raw = include_raw

    def invoke(self, messages: list) -> BaseModel | dict:
        entry = self._replay_state._next_entry(_hash_request(messages, self._schema.__name__))
        parsed = self._schema.model_validate(entry.response)
        if not self._include_raw:
            return parsed
        # Mimic the include_raw response shape. Legacy cassettes (no recorded
        # usage) yield raw=None and callers fall back to estimated usage.
        raw = SimpleNamespace(usage_metadata=entry.usage_metadata) if entry.usage_metadata is not None else None
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class ReplayChatModel:
    """Chat-model stand-in that replays a recorded cassette without network calls.

    In strict mode (default), a request whose fingerprint differs from the
    recorded one raises CassetteMissError — the pipeline's prompts have drifted
    and the cassette should be re-recorded. With strict=False the next recorded
    response is returned regardless, which is useful for evaluating downstream
    (post-LLM) logic changes that intentionally alter prompts.
    """

    def __init__(self, cassette_path: Path, strict: bool = True) -> None:
        cassette_path = Path(cassette_path)
        if not cassette_path.exists():
            raise CassetteMissError(
                f"Cassette file not found: {cassette_path}. Run the evaluation in record mode first."
            )
        self._cassette = Cassette.load(cassette_path)
        self._cassette_path = cassette_path
        self._strict = strict
        self._position = 0

    def with_structured_output(self, schema: type[BaseModel], include_raw: bool = False) -> _ReplayStructuredModel:
        return _ReplayStructuredModel(replay_state=self, schema=schema, include_raw=include_raw)

    def _next_entry(self, request_hash: str) -> CassetteEntry:
        if self._position >= len(self._cassette.entries):
            raise CassetteMissError(
                f"Cassette exhausted after {len(self._cassette.entries)} call(s): {self._cassette_path}. "
                "The pipeline now makes more LLM calls than when the cassette was recorded."
            )
        entry = self._cassette.entries[self._position]
        if self._strict and entry.request_hash != request_hash:
            raise CassetteMissError(
                f"Request hash mismatch at call {self._position + 1}: the pipeline's prompts have changed "
                f"since this cassette was recorded ({self._cassette_path}). "
                "Re-record the cassette, or use strict=False to replay by call order."
            )
        self._position += 1
        return entry
