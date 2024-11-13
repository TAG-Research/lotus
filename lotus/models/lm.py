from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from lotus.templates import task_instructions


class LM(ABC):
    """Abstract class for language models."""

    def _init__(self):
        pass

    @abstractmethod
    def count_tokens(self, prompt: Union[str, list]) -> int:
        """
        Counts the number of tokens in the given prompt after processing.

        Args:
            prompt (Union[str, list]): The prompt to count tokens for. This can be a string or a list of messages.

        Returns:
            int: The number of tokens in the prompt.
        """
        # Process the prompt to remove image data
        processed_prompt = self._remove_image_data(prompt)

        # Call the implementation-specific token counting method
        return self._count_tokens_impl(processed_prompt)

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

    def format_logprobs_for_cascade(self, logprobs: List) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Formats the logprobs for the cascade.

        Args:
            logprobs (List): The logprobs to format.

        Returns:
            Tuple[List[List[str]], List[List[float]]]: A tuple containing the tokens and their corresponding confidences.
        """
        pass

    @abstractmethod
    def __call__(
        self, messages_batch: Union[List, List[List]], **kwargs: Dict[str, Any]
    ) -> Union[List, Tuple[List, List]]:
        """Invoke the LLM.

        Args:
            messages_batch (Union[List, List[List]]): Either one prompt or a list of prompts in message format.
            kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify inference parameters.

        Returns:
            Union[List, Tuple[List, List]]: A list of outputs for each prompt in the batch. If logprobs is specified in the keyword arguments,
            then a list of logprobs is also returned.
        """
        pass
