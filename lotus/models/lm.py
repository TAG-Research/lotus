from typing import Any

import numpy as np
from litellm import batch_completion, completion_cost
from litellm.types.utils import ChatCompletionTokenLogprob, Choices, ModelResponse
from litellm.utils import token_counter
from tokenizers import Tokenizer

from lotus.types import LMOutput, LMStats, LogprobsForCascade, LogprobsForFilterCascade


class LM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_ctx_len: int = 128000,
        max_tokens: int = 512,
        tokenizer: Tokenizer | None = None,
        **kwargs: dict[str, Any],
    ):
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

        self.stats: LMStats = LMStats()

    def __call__(
        self, messages: list[dict[str, str]] | list[list[dict[str, str]]], **kwargs: dict[str, Any]
    ) -> LMOutput:
        kwargs_for_batch = self._format_batch_kwargs(kwargs)
        responses: list[ModelResponse] = batch_completion(
            self.model,
            messages,
            temperature=kwargs_for_batch.get("temperature"),
            max_tokens=kwargs_for_batch.get("max_tokens"),
            top_logprobs=kwargs_for_batch.get("top_logprobs"),
            logprobs=kwargs_for_batch.get("logprobs"),
        )
        outputs = [self._get_top_choice(resp) for resp in responses]
        logprobs = (
            [self._get_top_choice_logprobs(resp) for resp in responses] if kwargs_for_batch.get("logprobs") else None
        )

        for resp in responses:
            self._update_stats(resp)

        return LMOutput(outputs=outputs, logprobs=logprobs)

    def _update_stats(self, response: ModelResponse):
        if not hasattr(response, "usage"):
            return

        self.stats.total_usage.prompt_tokens += response.usage.prompt_tokens
        self.stats.total_usage.completion_tokens += response.usage.completion_tokens
        self.stats.total_usage.total_tokens += response.usage.total_tokens
        self.stats.total_usage.total_cost += completion_cost(completion_response=response)

    def _format_batch_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        all_kwargs = {**self.kwargs, **kwargs}
        if all_kwargs.get("logprobs", False):
            all_kwargs["top_logprobs"] = all_kwargs.get("top_logprobs", 10)
        return {k: v for k, v in all_kwargs.items() if k in ["temperature", "max_tokens", "top_logprobs", "logprobs"]}

    def _get_top_choice(self, response: ModelResponse) -> str:
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        if choice.message.content is None:
            raise ValueError(f"No content in response: {response}")
        return choice.message.content

    def _get_top_choice_logprobs(self, response: ModelResponse) -> list[ChatCompletionTokenLogprob]:
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        logprobs = choice.logprobs["content"]
        return [ChatCompletionTokenLogprob(**logprob) for logprob in logprobs]

    def format_logprobs_for_cascade(self, logprobs: list[list[ChatCompletionTokenLogprob]]) -> LogprobsForCascade:
        all_tokens = []
        all_confidences = []
        for resp_logprobs in logprobs:
            tokens = [logprob.token for logprob in resp_logprobs]
            confidences = [np.exp(logprob.logprob) for logprob in resp_logprobs]
            all_tokens.append(tokens)
            all_confidences.append(confidences)
        return LogprobsForCascade(tokens=all_tokens, confidences=all_confidences)

    def format_logprobs_for_filter_cascade(
        self, logprobs: list[list[ChatCompletionTokenLogprob]]
    ) -> LogprobsForFilterCascade:
        # Get base cascade format first
        base_cascade = self.format_logprobs_for_cascade(logprobs)
        all_true_probs = []

        def get_normalized_true_prob(token_probs: dict[str, float]) -> float | None:
            if "True" in token_probs and "False" in token_probs:
                true_prob = token_probs["True"]
                false_prob = token_probs["False"]
                return true_prob / (true_prob + false_prob)
            return None

        # Get true probabilities for filter cascade
        for resp_idx, response_logprobs in enumerate(logprobs):
            true_prob = None
            for logprob in response_logprobs:
                token_probs = {top.token: np.exp(top.logprob) for top in logprob.top_logprobs}
                true_prob = get_normalized_true_prob(token_probs)
                if true_prob is not None:
                    break

            # Default to 1 if "True" in tokens, 0 if not
            if true_prob is None:
                true_prob = 1 if "True" in base_cascade.tokens[resp_idx] else 0

            all_true_probs.append(true_prob)

        return LogprobsForFilterCascade(
            tokens=base_cascade.tokens, confidences=base_cascade.confidences, true_probs=all_true_probs
        )

    def count_tokens(self, messages: list[dict[str, str]] | str) -> int:
        """Count tokens in messages using either custom tokenizer or model's default tokenizer"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        custom_tokenizer: dict[str, Any] | None = None
        if self.tokenizer:
            custom_tokenizer = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer)

        # Pass values directly rather than using kwargs dict to preserve typing
        return token_counter(
            custom_tokenizer=custom_tokenizer,
            model=self.model,
            messages=messages,
        )

    def print_total_usage(self):
        print(f"Total cost: ${self.stats.total_usage.total_cost:.6f}")
        print(f"Total prompt tokens: {self.stats.total_usage.prompt_tokens}")
        print(f"Total completion tokens: {self.stats.total_usage.completion_tokens}")
        print(f"Total tokens: {self.stats.total_usage.total_tokens}")

    def reset_stats(self):
        self.stats = LMStats(
            total_usage=LMStats.TotalUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, total_cost=0.0)
        )
