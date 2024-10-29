import numpy as np
from litellm import batch_completion, token_counter
from litellm.types.utils import ChatCompletionTokenLogprob, ModelResponse

from lotus.types import LMOutput, LogprobsForCascade, LogprobsForFilterCascade


class LM:
    def __init__(self, model="gpt-4o-mini", temperature=0.0, max_ctx_len=128000, max_tokens=512, **kwargs):
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
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
