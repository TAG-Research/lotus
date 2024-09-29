import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff
import numpy as np
import openai
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer

import lotus
from lotus.models.lm import LM

ERRORS = (openai.RateLimitError, openai.APIError)


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class OpenAIModel(LM):
    """Wrapper around OpenAI, Databricks, and vLLM OpenAI server

    Args:
        model (str): The name of the model to use.
        api_key (Optional[str]): An API key (e.g. from OpenAI or Databricks).
        api_base (Optional[str]): The endpoint of the server.
        provider (str): Either openai, dbrx, or vllm.
        max_batch_size (int): The maximum batch size for the model.
        max_ctx_len (int): The maximum context length for the model.
        **kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify inference parameters.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        hf_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        provider: str = "openai",
        max_batch_size: int = 64,
        max_ctx_len: int = 4096,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.provider = provider
        self.use_chat = provider in ["openai", "dbrx", "ollama"]
        self.max_batch_size = max_batch_size
        self.max_ctx_len = max_ctx_len
        self.hf_name = hf_name if hf_name is not None else model

        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1,
            "n": 1,
            **kwargs,
        }

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "None")
        self.client = OpenAI(api_key=api_key if api_key else "None", base_url=api_base)

        # TODO: Refactor this
        if self.provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)

    def handle_chat_request(self, messages: List, **kwargs: Dict[str, Any]) -> Union[List, Tuple[List, List]]:
        """Handle single chat request to OpenAI server.

        Args:
            messages_batch (List): A prompt in message format.
            **kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify things such as the prompt, temperature,
                      model name, max tokens, etc.

        Returns:
            Union[List, Tuple[List, List]]: A list of outputs for each prompt in the batch (just one in this case). If logprobs is specified in the keyword arguments,
            then a list of logprobs is also returned.
        """
        if kwargs.get("logprobs", False):
            kwargs["top_logprobs"] = 1

        kwargs = {**self.kwargs, **kwargs}
        kwargs["messages"] = messages
        response = self.chat_request(**kwargs)

        choices = response["choices"]
        completions = [c["message"]["content"] for c in choices]

        if kwargs.get("logprobs", False):
            logprobs = [c["logprobs"] for c in choices]
            return completions, logprobs

        return completions

    def handle_completion_request(self, messages: List, **kwargs):
        """Handle a potentially batched completions request to OpenAI server.

        Args:
            messages_batch: A list of prompts in message format.
            **kwargs: Additional keyword arguments. They can be used to specify things such as the prompt, temperature,
                      model name, max tokens, etc.

        Returns:
            Union[List, Tuple[List, List]]: A list of outputs for each prompt in the batch. If logprobs is specified in the keyword arguments,
            then a list of logprobs is also returned.
        """
        if not isinstance(messages[0], list):
            prompt = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        else:
            prompt = [
                self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]

        kwargs = {**self.kwargs, **kwargs}
        kwargs["prompt"] = prompt
        response = self.completion_request(**kwargs)

        choices = response["choices"]
        completions = [c["text"] for c in choices]

        if kwargs.get("logprobs", False):
            logprobs = [c["logprobs"] for c in choices]
            return completions, logprobs

        return completions

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, messages: List, **kwargs) -> Union[List, Tuple[List, List]]:
        """Handle single request to OpenAI server. Decides whether chat or completion endpoint is necessary.

        Args:
            messages_batch: A prompt in message format.
            **kwargs: Additional keyword arguments. They can be used to specify things such as the prompt, temperature,
                      model name, max tokens, etc.

        Returns:
            A list of text outputs for each prompt in the batch (just one in this case).
            If logprobs is specified in the keyword arguments, hen a list of logprobs is also returned (also of size one).
        """
        if self.use_chat:
            return self.handle_chat_request(messages, **kwargs)
        else:
            return self.handle_completion_request(messages, **kwargs)

    def batched_chat_request(self, messages_batch: List, **kwargs) -> Union[List, Tuple[List, List]]:
        """Handle batched chat request to OpenAI server.

        Args:
            messages_batch (List): Either one prompt or a list of prompts in message format.
            **kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify inference parameters.

        Returns:
            Union[List, Tuple[List, List]]: A list of outputs for each prompt in the batch. If logprobs is specified in the keyword arguments,
            then a list of logprobs is also returned.
        """

        batch_size = len(messages_batch)
        text_ret = [None] * batch_size
        logprobs_ret = [None] * batch_size
        threads = []

        def thread_function(idx, messages, kwargs):
            text = self(messages, **kwargs)
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

    def __call__(
        self, messages_batch: Union[List, List[List]], **kwargs: Dict[str, Any]
    ) -> Union[List, Tuple[List, List]]:
        lotus.logger.debug(f"OpenAIModel.__call__ messages_batch: {messages_batch}")
        lotus.logger.debug(f"OpenAIModel.__call__ kwargs: {kwargs}")
        # Bakes max batch size into model call. # TODO: Figure out less hacky way to do this.
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

        if self.use_chat and isinstance(messages_batch[0], list):
            return self.batched_chat_request(messages_batch, **kwargs)

        return self.request(messages_batch, **kwargs)

    def count_tokens(self, prompt: Union[str, list]) -> int:
        if isinstance(prompt, str):
            if self.provider != "openai":
                return len(self.tokenizer(prompt)["input_ids"])

            return len(self.tokenizer.encode(prompt))
        else:
            if self.provider != "openai":
                return len(self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))

            return sum(len(self.tokenizer.encode(message["content"])) for message in prompt)

    def format_logprobs_for_cascade(self, logprobs: List) -> Tuple[List[List[str]], List[List[float]]]:
        all_tokens = []
        all_confidences = []
        for idx in range(len(logprobs)):
            if self.provider == "vllm":
                tokens = logprobs[idx]["tokens"]
                confidences = np.exp(logprobs[idx]["token_logprobs"])
            elif self.provider == "openai":
                content = logprobs[idx]["content"]
                tokens = [content[t_idx]["token"] for t_idx in range(len(content))]
                confidences = np.exp([content[t_idx]["logprob"] for t_idx in range(len(content))])
            all_tokens.append(tokens)
            all_confidences.append(confidences)

        return all_tokens, all_confidences

    def chat_request(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Send chat request to OpenAI server.

        Args:
            **kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify things such as the prompt, temperature,
                      model name, max tokens, etc.

        Returns:
            dict: OpenAI chat completion response.
        """
        return self.client.chat.completions.create(**kwargs).model_dump()

    def completion_request(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Send completion request to OpenAI server.

        Args:
            **kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify things such as the prompt, temperature,
                      model name, max tokens, etc.

        Returns:
            dict: OpenAI completion response.
        """
        return self.client.completions.create(**kwargs).model_dump()

    @property
    def max_tokens(self) -> int:
        return self.kwargs["max_tokens"]
