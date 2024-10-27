from abc import ABC, abstractmethod
from typing import Any


class LM(ABC):
    """Abstract class for language models."""

    def _init__(self):
        pass

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

    @abstractmethod
    def __call__(
        self, messages_batch: list | list[list], **kwargs: dict[str, Any]
    ) -> list | tuple[list[list[str]], list[list[float]]]:
        """Invoke the LLM.

        Args:
            messages_batch (list | list[list]): Either one prompt or a list of prompts in message format.
            kwargs (dict[str, Any]): Additional keyword arguments. They can be used to specify inference parameters.

        Returns:
            list | tuple[list[list[str]], list[list[float]]]: A list of outputs for each prompt in the batch. If logprobs is specified in the keyword arguments,
            then a list of logprobs is also returned.
        """
        pass
