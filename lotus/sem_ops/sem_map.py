from typing import Callable

import pandas as pd

import lotus
from lotus.templates import task_instructions
from lotus.types import SemanticMapOutput, SemanticMapPostprocessOutput

from .postprocessors import map_postprocess


def sem_map(
    docs: list[str],
    model: lotus.models.LM,
    user_instruction: str,
    postprocessor: Callable[[list[str], bool], SemanticMapPostprocessOutput] = map_postprocess,
    examples_df_txt: list[str] | None = None,
    examples_answers: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> SemanticMapOutput:
    """
    Maps a list of documents to a list of outputs using a model.

    Args:
        docs (list[str]): The list of documents to map.
        model (lotus.models.LM): The model to use.
        user_instruction (str): The user instruction for map.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to map_postprocess.
        examples_df_txt (list[str] | None): The text for examples. Defaults to None.
        examples_answers (list[str] | None): The answers for examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.

    Returns:
        SemanticMapOutput: The outputs, raw outputs, and explanations.
    """
    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.map_formatter(
            doc, user_instruction, examples_df_txt, examples_answers, cot_reasoning, strategy=strategy
        )
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # call model
    raw_outputs = model(inputs)
    assert isinstance(raw_outputs, list) and all(
        isinstance(item, str) for item in raw_outputs
    ), "Model must return a list of strings"

    # post process results
    postprocess_output = postprocessor(raw_outputs, strategy in ["cot", "zs-cot"])
    lotus.logger.debug(f"raw_outputs: {raw_outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")

    return SemanticMapOutput(**postprocess_output.model_dump())


@pd.api.extensions.register_dataframe_accessor("sem_map")
class SemMapDataframe:
    """DataFrame accessor for semantic map."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        user_instruction: str,
        postprocessor: Callable[[list[str], bool], SemanticMapPostprocessOutput] = map_postprocess,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        suffix: str = "_map",
        examples: pd.DataFrame | None = None,
        strategy: str | None = None,
    ) -> pd.DataFrame:
        """
        Applies semantic map over a dataframe.

        Args:
            user_instruction (str): The user instruction for map.
            postprocessor (Callable): The postprocessor for the model outputs. Defaults to map_postprocess.
            return_explanations (bool): Whether to return explanations. Defaults to False.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.
            suffix (str): The suffix for the new columns. Defaults to "_map".
            examples (pd.DataFrame | None): The examples dataframe. Defaults to None.
            strategy (str | None): The reasoning strategy. Defaults to None.

        Returns:
            pd.DataFrame: The dataframe with the new mapped columns.
        """
        col_li = lotus.nl_expression.parse_cols(user_instruction)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        df_txt = task_instructions.df2text(self._obj, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_df_txt = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_df_txt = task_instructions.df2text(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot":
                return_explanations = True
                cot_reasoning = examples["Reasoning"].tolist()

        output = sem_map(
            df_txt,
            lotus.settings.lm,
            formatted_usr_instr,
            postprocessor=postprocessor,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
        )

        new_df = self._obj.copy()
        new_df[suffix] = output.outputs
        if return_explanations:
            new_df["explanation" + suffix] = output.explanations
        if return_raw_outputs:
            new_df["raw_output" + suffix] = output.raw_outputs

        return new_df
