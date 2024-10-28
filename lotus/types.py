from typing import Any

from pydantic import BaseModel


class StatsMixin(BaseModel):
    stats: dict[str, Any] | None = None


# TODO: Figure out better logprobs type
class LogprobsMixin(BaseModel):
    logprobs: list[dict[str, Any]] | None = None


class SemanticMapPostprocessOutput(StatsMixin, LogprobsMixin):
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


class SemanticMapOutput(SemanticMapPostprocessOutput):
    pass


class SemanticFilterPostprocessOutput(BaseModel):
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]


class SemanticFilterOutput(SemanticFilterPostprocessOutput, StatsMixin, LogprobsMixin):
    pass


class SemanticAggOutput(BaseModel):
    outputs: list[str]


class SemanticExtractPostprocessOutput(BaseModel):
    raw_outputs: list[str]
    outputs: list[str]
    quotes: list[str | None]


class SemanticExtractOutput(SemanticExtractPostprocessOutput):
    pass


class SemanticJoinOutput(StatsMixin):
    join_results: list[tuple[int, int, str | None]]
    filter_outputs: list[bool]
    all_raw_outputs: list[str]
    all_explanations: list[str | None]


class SemanticTopKOutput(StatsMixin):
    indexes: list[int]
