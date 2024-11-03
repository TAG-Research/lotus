import numpy as np
import ujson
import os
from typing import Any
import litellm
from litellm import batch_completion, token_counter
from litellm.caching import Cache
from litellm.types.utils import ChatCompletionTokenLogprob, ModelResponse
import logging
import functools
from functools import lru_cache
from lotus.types import LMOutput, LogprobsForCascade, LogprobsForFilterCascade

litellm.cache = Cache(disk_cache_dir=".lotus_cache", type="disk")

class LM:
    def __init__(self, model="gpt-4o-mini", temperature=0.0, max_ctx_len=128000, cache=True, max_tokens=512, **kwargs):
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self.api_calls = 0
        self.cache = cache
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _messages_to_cache_key(self, messages):
        if isinstance(messages[0], list):
            return tuple(tuple(frozenset(m.items()) for m in msg) for msg in messages)
        return tuple(frozenset(m.items()) for m in messages)

    def _cache_key_to_messages(self, messages_tuple):
        if isinstance(messages_tuple[0], tuple):
            return [[dict(m) for m in msg] for msg in messages_tuple]
        return [dict(m) for m in messages_tuple]

    @lru_cache(maxsize=128)
    def _cached_completion(self, messages_tuple, **kwargs):
        messages = self._cache_key_to_messages(messages_tuple)
        return self._batch_complete(messages, **kwargs)

    def _batch_complete(self, messages, **kwargs):
        """Execute batch completion with given parameters."""
        return batch_completion(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_logprobs=kwargs.get("top_logprobs"),
            logprobs=kwargs.get("logprobs"),
        )
    
    def __call__(self, messages: list[dict[str, str]] | list[list[dict[str, str]]], **kwargs: dict[str, Any]
    ) -> LMOutput:
        kwargs = {**self.kwargs, **kwargs}
        kwargs_for_batch = self._format_batch_kwargs(kwargs)
        cache = kwargs.pop("cache", self.cache)
        if kwargs.get("logprobs", False):
            kwargs["top_logprobs"] = kwargs.get("top_logprobs", 10)

        if cache:
            messages_tuple = self._messages_to_cache_key(messages)
            responses = self._cached_completion(messages_tuple, **kwargs_for_batch)
        else:
            responses = self._batch_complete(messages, **kwargs_for_batch)
            self.api_calls += 1
        self.logger.info(f"Making API call #{self.api_calls}")
        outputs = [self._get_top_choice(resp) for resp in responses]
        logprobs = [self._get_top_choice_logprobs(resp) for resp in responses] if kwargs.get("logprobs") else None

        return LMOutput(outputs=outputs, logprobs=logprobs)

    def _get_top_choice(self, response: ModelResponse) -> str:
        return response.choices[0].message.content

    def _get_top_choice_logprobs(self, response: ModelResponse) -> list[ChatCompletionTokenLogprob]:
        logprobs = response.choices[0].logprobs["content"]
        return [ChatCompletionTokenLogprob(**logprob) for logprob in logprobs]

    def format_logprobs_for_cascade(self, logprobs: list[list[ChatCompletionTokenLogprob]]) -> LogprobsForCascade:
        all_tokens = []
        all_confidences = []
        for resp in range(len(logprobs)):
            tokens = [logprob.token for logprob in logprobs[resp]]
            confidences = [np.exp(logprob.logprob) for logprob in logprobs[resp]]
            all_tokens.append(tokens)
            all_confidences.append(confidences)
        return LogprobsForCascade(tokens=all_tokens, confidences=all_confidences)

    def format_logprobs_for_filter_cascade(
        self, logprobs: list[list[ChatCompletionTokenLogprob]]
    ) -> LogprobsForFilterCascade:
        all_tokens = []
        all_confidences = []
        all_true_probs = []

        for resp in range(len(logprobs)):
            all_tokens.append([logprob.token for logprob in logprobs[resp]])
            all_confidences.append([np.exp(logprob.logprob) for logprob in logprobs[resp]])
            top_logprobs = {x.token: np.exp(x.logprob) for x in logprobs[resp]}
            true_prob, false_prob = 0, 0
            if top_logprobs and "True" in top_logprobs and "False" in top_logprobs:
                true_prob = np.exp(top_logprobs["True"])
                false_prob = np.exp(top_logprobs["False"])
                all_true_probs.append(true_prob / (true_prob + false_prob))
            else:
                all_true_probs.append(1 if "True" in top_logprobs else 0)
        return LogprobsForFilterCascade(tokens=all_tokens, confidences=all_confidences, true_probs=all_true_probs)

    def count_tokens(self, messages: list[dict[str, str]] | str) -> int:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return token_counter(model=self.model, messages=messages)
    
    def _format_batch_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        all_kwargs = {**self.kwargs, **kwargs}
        if all_kwargs.get("logprobs", False):
            all_kwargs["top_logprobs"] = all_kwargs.get("top_logprobs", 10)
        return {k: v for k, v in all_kwargs.items() if k in ["temperature", "max_tokens", "top_logprobs", "logprobs"]}
