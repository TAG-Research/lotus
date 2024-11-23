from typing import Callable

import pandas as pd

import lotus
from lotus.models import LM
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticExtractOutput, SemanticExtractPostprocessOutput

from .postprocessors import extract_postprocess


def sem_extract(
    docs: list[str],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str]], SemanticExtractPostprocessOutput] = extract_postprocess,
) -> SemanticExtractOutput:
    """
    Extracts attributes and values from a list of documents using a model.

    Args:
        docs (list[str]): The list of documents to extract from.
        model (lotus.models.LM): The model to use.
        output_cols (dict[str, str | None]): A mapping from desired output column names to optional descriptions.
        extract_quotes (bool): Whether to extract quotes for the output columns. Defaults to False.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to extract_postprocess.

    Returns:
        SemanticExtractOutput: The outputs, raw outputs, and quotes.
    """

    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = task_instructions.extract_formatter(doc, output_cols, extract_quotes)
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # call model
    lm_output: LMOutput = model(inputs, response_format={"type": "json_object"})

    # post process results
    postprocess_output = postprocessor(lm_output.outputs)
    lotus.logger.debug(f"raw_outputs: {lm_output.outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")

    return SemanticExtractOutput(**postprocess_output.model_dump())


@pd.api.extensions.register_dataframe_accessor("sem_extract")
class SemExtractDataFrame:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        input_cols: list[str],
        output_cols: dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable[[list[str]], SemanticExtractPostprocessOutput] = extract_postprocess,
        return_raw_outputs: bool = False,
    ) -> pd.DataFrame:
        """
        Extracts the attributes and values of a dataframe.

        Args:
            input_cols (list[str]): The columns that a model should extract from.
            output_cols (dict[str, str | None]): A mapping from desired output column names to optional descriptions.
            extract_quotes (bool): Whether to extract quotes for the output columns. Defaults to False.
            postprocessor (Callable): The postprocessor for the model outputs. Defaults to extract_postprocess.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.

        Returns:
            pd.DataFrame: The dataframe with the new mapped columns.
        """

        # check that column exists
        for column in input_cols:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        docs = task_instructions.df2text(self._obj, input_cols)

        out = sem_extract(
            docs=docs,
            model=lotus.settings.lm,
            output_cols=output_cols,
            extract_quotes=extract_quotes,
            postprocessor=postprocessor,
        )

        new_df = self._obj.copy()
        for i, output_dict in enumerate(out.outputs):
            for key, value in output_dict.items():
                if key not in new_df.columns:
                    new_df[key] = None
                new_df.loc[i, key] = value

        if return_raw_outputs:
            new_df["raw_output"] = out.raw_outputs

        new_df = new_df.reset_index(drop=True)

        return new_df
