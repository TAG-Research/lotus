from typing import Any, Callable

import pandas as pd

import lotus
from lotus.templates import task_instructions
from lotus.types import SemanticExtractOutput, SemanticExtractPostprocessOutput

from .postprocessors import extract_postprocess


def sem_extract(
    docs: list[str],
    model: lotus.models.LM,
    user_instruction: str,
    postprocessor: Callable[[list[str]], SemanticExtractPostprocessOutput] = extract_postprocess,
) -> SemanticExtractOutput:
    """
    Extracts from a list of documents using a model.

    Args:
        docs (list[str]): The list of documents to extract from.
        model (lotus.models.LM): The model to use.
        user_instruction (str): The user instruction for extract.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to extract_postprocess.

    Returns:
        SemanticExtractOutput: The outputs, raw outputs, and quotes.
    """
    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.extract_formatter(doc, user_instruction)
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # call model
    raw_outputs = model(inputs)
    if isinstance(raw_outputs, tuple):
        raw_outputs, _ = raw_outputs
    else:
        assert isinstance(raw_outputs, list)

    # post process results
    postprocess_output = postprocessor(raw_outputs)
    lotus.logger.debug(f"raw_outputs: {raw_outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"quotes: {postprocess_output.quotes}")

    return SemanticExtractOutput(**postprocess_output.model_dump())


@pd.api.extensions.register_dataframe_accessor("sem_extract")
class SemExtractDataframe:
    """DataFrame accessor for semantic extract."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        user_instruction: str,
        postprocessor: Callable[[list[str]], SemanticExtractPostprocessOutput] = extract_postprocess,
        return_raw_outputs: bool = False,
        suffix: str = "_extract",
    ) -> pd.DataFrame:
        """
        Applies semantic extract over a dataframe.

        Args:
            user_instruction (str): The user instruction for extract.
            postprocessor (Callable): The postprocessor for the model outputs. Defaults to extract_postprocess.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.
            suffix (str): The suffix for the new columns. Defaults to "_extract".
        Returns:
            pd.DataFrame: The dataframe with the new extracted values.
        """
        col_li = lotus.nl_expression.parse_cols(user_instruction)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        df_txt = task_instructions.df2text(self._obj, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        output = sem_extract(
            df_txt,
            lotus.settings.lm,
            formatted_usr_instr,
            postprocessor=postprocessor,
        )

        new_df = self._obj
        new_df["answers" + suffix] = output.outputs
        new_df["quotes" + suffix] = output.quotes
        if return_raw_outputs:
            new_df["raw_output" + suffix] = output.raw_outputs

        return new_df
