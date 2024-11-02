import numpy as np
from litellm import batch_completion, token_counter
from litellm.types.utils import ChatCompletionTokenLogprob, ModelResponse
from tokenizers import Tokenizer

from lotus.types import LMOutput, LogprobsForCascade, LogprobsForFilterCascade


class LM:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_ctx_len: int = 128000, max_tokens: int = 512, tokenizer: Tokenizer = None, **kwargs):
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    def __call__(self, messages=None, **kwargs) -> LMOutput:
        kwargs = {**self.kwargs, **kwargs}
        if kwargs.get("logprobs", False):
            kwargs["top_logprobs"] = kwargs.get("top_logprobs", 10)

        responses: list[ModelResponse] = batch_completion(model=self.model, messages=messages, **kwargs)
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
            tokens=base_cascade.tokens,
            confidences=base_cascade.confidences,
            true_probs=all_true_probs
        )

    def count_tokens(self, messages: list[dict[str, str]] | str) -> int:
        """Count tokens in messages using either custom tokenizer or model's default tokenizer"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            
        kwargs = {"model": self.model, "messages": messages}
        if self.tokenizer:
            kwargs["custom_tokenizer"] = {
                "type": "huggingface_tokenizer", 
                "tokenizer": self.tokenizer
            }
            
        return token_counter(**kwargs)
