from typing import Any

import pandas as pd
from litellm.types.utils import ChatCompletionTokenLogprob
from pydantic import BaseModel


################################################################################
# Mixins
################################################################################
class StatsMixin(BaseModel):
    stats: dict[str, Any] | None = None


class LogprobsMixin(BaseModel):
    # for each response, we have a list of tokens, and for each token, we have a ChatCompletionTokenLogprob
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


################################################################################
# LM related
################################################################################
class LMOutput(LogprobsMixin):
    outputs: list[str]


class LMStats(BaseModel):
    class TotalUsage(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        total_cost: float = 0.0
        cache_hits: int = 0

    total_usage: TotalUsage = TotalUsage()


class LogprobsForCascade(BaseModel):
    tokens: list[list[str]]
    confidences: list[list[float]]


class LogprobsForFilterCascade(LogprobsForCascade):
    true_probs: list[float]


################################################################################
# Semantic operation outputs
################################################################################
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


class SemJoinCascadeArgs(BaseModel):
    recall_target: float | None = None
    precision_target: float | None = None
    sampling_percentage: float = 0.1
    failure_probability: float = 0.2
    map_instruction: str | None = None
    map_examples: pd.DataFrame | None = None

    # to enable pandas
    class Config:
        arbitrary_types_allowed = True

class SemanticTopKOutput(StatsMixin):
    indexes: list[int]


################################################################################
# RM related
################################################################################
class RMOutput(BaseModel):
    distances: list[list[float]]
    indices: list[list[int]]


################################################################################
# Reranker related
################################################################################
class RerankerOutput(BaseModel):
    indices: list[int]
