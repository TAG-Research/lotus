import hashlib
from typing import Any

import litellm
import numpy as np
from litellm import batch_completion, completion_cost
from litellm.types.utils import ChatCompletionTokenLogprob, Choices, ModelResponse
from litellm.utils import token_counter
from openai import OpenAIError
from tokenizers import Tokenizer

import lotus
from lotus.cache import Cache
from lotus.types import LMOutput, LMStats, LogprobsForCascade, LogprobsForFilterCascade
from lotus.templates import task_instructions


class LM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_ctx_len: int = 128000,
        max_tokens: int = 512,
        max_batch_size: int = 64,
        tokenizer: Tokenizer | None = None,
        max_cache_size: int = 1024,
        **kwargs: dict[str, Any],
    ):
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

        self.stats: LMStats = LMStats()
        self.cache = Cache(max_cache_size)

    def __call__(self, messages: list[list[dict[str, str]]], **kwargs: dict[str, Any]) -> LMOutput:
        all_kwargs = {**self.kwargs, **kwargs}

        # Set top_logprobs if logprobs requested
        if all_kwargs.get("logprobs", False):
            all_kwargs.setdefault("top_logprobs", 10)

        # Check cache and separate cached and uncached messages
        hashed_messages = [self._hash_messages(msg, all_kwargs) for msg in messages]
        cached_responses = [self.cache.get(hash) for hash in hashed_messages]
        uncached_data = [
            (msg, hash) for msg, hash, resp in zip(messages, hashed_messages, cached_responses) if resp is None
        ]
        self.stats.total_usage.cache_hits += len(messages) - len(uncached_data)

        # Process uncached messages in batches
        uncached_responses = self._process_uncached_messages(uncached_data, all_kwargs)

        # Add new responses to cache
        for resp, (_, hash) in zip(uncached_responses, uncached_data):
            self._cache_response(resp, hash)

        # Merge all responses in original order and extract outputs
        all_responses = self._merge_responses(cached_responses, uncached_responses)
        outputs = [self._get_top_choice(resp) for resp in all_responses]
        logprobs = (
            [self._get_top_choice_logprobs(resp) for resp in all_responses] if all_kwargs.get("logprobs") else None
        )

        return LMOutput(outputs=outputs, logprobs=logprobs)

    def _process_uncached_messages(self, uncached_data, all_kwargs):
        """Processes uncached messages in batches and returns responses."""
        uncached_responses = []
        for i in range(0, len(uncached_data), self.max_batch_size):
            batch = [msg for msg, _ in uncached_data[i : i + self.max_batch_size]]
            uncached_responses.extend(batch_completion(self.model, batch, drop_params=True, **all_kwargs))
        return uncached_responses

    def _cache_response(self, response, hash):
        """Caches a response and updates stats if successful."""
        if isinstance(response, OpenAIError):
            raise response
        self._update_stats(response)
        self.cache.insert(hash, response)

    def _hash_messages(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        """Hash messages and kwargs to create a unique key for the cache"""
        to_hash = str(self.model) + str(messages) + str(kwargs)
        return hashlib.sha256(to_hash.encode()).hexdigest()

    def _merge_responses(
        self, cached_responses: list[ModelResponse | None], uncached_responses: list[ModelResponse]
    ) -> list[ModelResponse]:
        """Merge cached and uncached responses, maintaining order"""
        uncached_iter = iter(uncached_responses)
        return [resp if resp is not None else next(uncached_iter) for resp in cached_responses]

    def _update_stats(self, response: ModelResponse):
        if not hasattr(response, "usage"):
            return

        self.stats.total_usage.prompt_tokens += response.usage.prompt_tokens
        self.stats.total_usage.completion_tokens += response.usage.completion_tokens
        self.stats.total_usage.total_tokens += response.usage.total_tokens

        try:
            self.stats.total_usage.total_cost += completion_cost(completion_response=response)
        except litellm.exceptions.NotFoundError as e:
            # Sometimes the model's pricing information is not available
            lotus.logger.debug(f"Error updating completion cost: {e}")

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
            processed_messages = self._remove_image_data(messages) # need to do same if not a string but dict
            messages = [{"role": "user", "content": processed_messages}]

        custom_tokenizer: dict[str, Any] | None = None
        if self.tokenizer:
            custom_tokenizer = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer)

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
        print(f"Total cache hits: {self.stats.total_usage.cache_hits}")

    def reset_stats(self):
        self.stats = LMStats(
            total_usage=LMStats.TotalUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, total_cost=0.0)
        )

    def reset_cache(self, max_size: int | None = None):
        self.cache.reset(max_size)

    def _remove_image_data(self, prompt: Union[str, list]) -> Union[str, list]:
        """
        Removes image data from the prompt.

        Args:
            prompt (Union[str, list]): The original prompt.

        Returns:
            Union[str, list]: The prompt without image data.
        """
        if isinstance(prompt, str):
            _, prompt_without_images = task_instructions.extract_image_data(prompt)
            return prompt_without_images
        elif isinstance(prompt, list):
            return [task_instructions.extract_image_data(p)[1] for p in prompt]
        else:
            raise ValueError("Prompt must be either a string or a list of strings.")
