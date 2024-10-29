from typing import Any

from litellm.types.utils import ChatCompletionTokenLogprob
from pydantic import BaseModel


class StatsMixin(BaseModel):
    stats: dict[str, Any] | None = None


# TODO: Figure out better logprobs type
class LogprobsMixin(BaseModel):
    # for each response, we have a list of tokens, and for each token, we have a ChatCompletionTokenLogprob
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


class SemanticMapPostprocessOutput(BaseModel):
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


class LMOutput(LogprobsMixin):
    outputs: list[str]


class LogprobsForCascade(BaseModel):
    tokens: list[list[str]]
    confidences: list[list[float]]


class LogprobsForFilterCascade(LogprobsForCascade):
    true_probs: list[float]
