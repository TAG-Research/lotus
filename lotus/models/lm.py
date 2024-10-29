import os
import functools
import threading
import ujson
from typing import Any
from abc import ABC, abstractmethod

import litellm
from litellm.caching import Cache


litellm.cache = Cache(disk_cache_dir=".lotus_cache", type="disk")

class LM:
    """Abstract class for language models."""

    def __init__(
        self,
        model: str,
        model_type: str = 'chat',
        temperature: float = 0.0,
        cache: bool = True,
        max_tokens: int = 512,
        max_ctx_len: int = 4096,
        **kwargs: dict[str, Any]
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.max_ctx_len = max_ctx_len
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        self.history = []

    def __call__(
        self, 
        messages_batch: list | list[list], 
        **kwargs: dict[str, Any]
    ) -> list[str] | tuple[list[str], list[dict[str, Any]]]:
        # Handle batch size limitation
        if isinstance(messages_batch[0], list) and len(messages_batch) > self.max_batch_size:
            text_ret = []
            logprobs_ret = []
            for i in range(0, len(messages_batch), self.max_batch_size):
                res = self(messages_batch[i : i + self.max_batch_size], **kwargs)
                if kwargs.get("logprobs", False):
                    text, logprobs = res
                    logprobs_ret.extend(logprobs)
                else:
                    text = res
                text_ret.extend(text)

            if kwargs.get("logprobs", False):
                return text_ret, logprobs_ret
            return text_ret

        # Handle chat vs completion
        if self.model_type == "chat" and isinstance(messages_batch[0], list):
            return self.batched_chat_request(messages_batch, **kwargs)

        return self.request(messages_batch, **kwargs)

    def request(
        self, 
        messages: list | list[list], 
        **kwargs: dict[str, Any]
    ) -> list | tuple[list[list[str]], list[list[float]]]:
        cache = kwargs.pop("cache", self.cache)
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            completion = self.cached_litellm_completion if cache else self.litellm_completion
        else:
            completion = self.ached_litellm_text_completion if cache else self.litellm_text_completion

        if not isinstance(messages[0], list):
            messages = [messages]

        responses = []
        logprobs = []
        
        for msg in messages:
            response = completion(ujson.dumps({
                "model": self.model,
                "messages": msg if self.model_type == "chat" else None,
                "prompt": msg[0]['content'] if self.model_type != "chat" else None,
                **kwargs
            }))

            outputs = [c.message.content if hasattr(c, "message") else c["text"] 
                      for c in response.choices]
            responses.extend(outputs)

            if kwargs.get("logprobs"):
                logprobs.extend([c.get("logprobs") for c in response.choices])

            self.log_history(msg, response, outputs, kwargs)

        if kwargs.get("logprobs"):
            return responses, logprobs
        return responses

    def batched_chat_request(
        self, 
        messages_batch: list, 
        **kwargs: dict[str, Any]
    ) -> list | tuple[list[list[str]], list[list[float]]]:
        batch_size = len(messages_batch)
        text_ret = [None] * batch_size
        logprobs_ret = [None] * batch_size
        threads = []

        def thread_function(idx, messages, kwargs):
            text = self.request(messages, **kwargs)
            if kwargs.get("logprobs", False):
                text, logprobs = text
                logprobs_ret[idx] = logprobs[0]
            text_ret[idx] = text[0]

        for idx, messages in enumerate(messages_batch):
            thread = threading.Thread(target=thread_function, args=(idx, messages, kwargs))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if kwargs.get("logprobs", False):
            return text_ret, logprobs_ret

        return text_ret

    def log_history(self, messages, response, outputs, kwargs):
        entry = {
            "messages": messages,
            "kwargs": {k:v for k,v in kwargs.items() if not k.startswith("api_")},
            "response": response,
            "outputs": outputs,
            "usage": dict(response["usage"]),
            "cost": response["_hidden_params"].get("response_cost")
        }
        self.history.append(entry)


    @abstractmethod
    def count_tokens(self, prompt: str | list) -> int:
        """
        Counts the number of tokens in the given prompt.

        Args:
            prompt (str | list): The prompt to count tokens for. This can be a string or a list of messages.

        Returns:
            int: The number of tokens in the prompt.
        """
        pass

    def format_logprobs_for_cascade(self, logprobs: list) -> tuple[list[list[str]], list[list[float]]]:
        """
        Formats the logprobs for the cascade.

        Args:
            logprobs (list): The logprobs to format.

        Returns:
            tuple[list[list[str]], list[list[float]]]: A tuple containing the tokens and their corresponding confidences.
        """
        pass

    # @property
    # @abstractmethod
    # def max_ctx_len(self) -> int:
    #     """The maximum context length of the LLM."""
    #     pass

    # @property
    # @abstractmethod
    # def max_tokens(self) -> int:
    #     """The maximum number of tokens that can be generated by the LLM."""
    #     pass

    @functools.lru_cache(maxsize=None)
    def cached_litellm_completion(self,request):
        return self.litellm_completion(request, cache={"no-cache": False, "no-store": False})

    def litellm_completion(request, cache={"no-cache": True, "no-store": True}):
        kwargs = ujson.loads(request)
        return litellm.completion(cache=cache, **kwargs)

    @functools.lru_cache(maxsize=None)
    def cached_litellm_text_completion(self, request):
        return self.litellm_text_completion(request, cache=None)
 
    def litellm_text_completion(request, cache={"no-cache": True, "no-store": True}):
        kwargs = ujson.loads(request)

        # Extract the provider and model from the model string.
        model = kwargs.pop("model").split("/", 1)
        provider, model = model[0] if len(model) > 1 else "openai", model[-1]

        # Use the API key and base from the kwargs, or from the environment.
        api_key = kwargs.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
        api_base = kwargs.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

        # Build the prompt from the messages.
        prompt = '\n\n'.join([x['content'] for x in kwargs.pop("messages")] + ['BEGIN RESPONSE:'])

        return litellm.text_completion(cache=cache, model=f'text-completion-openai/{model}', api_key=api_key,
                                    api_base=api_base, prompt=prompt, **kwargs)


    def _green(text: str, end: str = "\n"):
        return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end

    def _red(text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def _inspect_history(self,lm, n: int = 1):
        """Prints the last n prompts and their completions."""

        for item in reversed(lm.history[-n:]):
            messages = item["messages"] or [{"role": "user", "content": item['prompt']}]
            outputs = item["outputs"]

            print("\n\n\n")
            for msg in messages:
                print(self._red(f"{msg['role'].capitalize()} message:"))
                print(msg['content'].strip())
                print("\n")

            print(self._red(f"Response:"))
            print(self._green(outputs[0].strip()))

            if len(outputs) > 1:
                choices_text = f" \t (and {len(outputs)-1} other completions)"
                print(self._red(choices_text, end=""))
            
        print("\n\n\n")