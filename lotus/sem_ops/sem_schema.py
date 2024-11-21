from typing import Callable

import pandas as pd

import lotus
from lotus.models import LM
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticSchemaOutput, SemanticSchemaPostprocessOutput

from .postprocessors import schema_postprocess


def sem_schema(
    docs: list[str],
    model: LM,
    columns: list[str],
    col_description: list[str],
    postprocessor: Callable[[list[str]], SemanticSchemaPostprocessOutput] = schema_postprocess,
) -> SemanticSchemaOutput:
    """
    Schemas a list of documents using a model.

    Args:
        docs (list[str]): The list of documents to schema.
        model (lotus.models.LM): The model to use.
        columns (list[str]): The columns to schema.
        col_description (str): The description of the columns.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to schema_postprocess.

    Returns:
        SemanticSchemaOutput: The outputs, raw outputs, and quotes.
    """

    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = task_instructions.schema_formatter(doc, columns, col_description)
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # call model
    lm_output: LMOutput = model(inputs)

    # post process results
    postprocess_output = postprocessor(lm_output.outputs)
    lotus.logger.debug(f"raw_outputs: {lm_output.outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")

    return SemanticSchemaOutput(**postprocess_output.model_dump())


@pd.api.extensions.register_dataframe_accessor("sem_schema")
class SemSchemaDataFrame:
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
        columns: list[str],
        col_description: list[str],
        postprocessor: Callable[[list[str]], SemanticSchemaPostprocessOutput] = schema_postprocess,
        return_raw_outputs: bool = False,
    ) -> pd.DataFrame:
        """
        Schemas the attributes and values of a dataframe.

        Args:
            user_instruction (str): The columns from the documents to schema.
            columns (list[str]): The columns to schema.
            col_description (str): The description of the columns.
            postprocessor (Callable): The postprocessor for the model outputs. Defaults to schema_postprocess.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.

        Returns:
            pd.DataFrame: The dataframe with the new mapped columns.
        """
        col_li = lotus.nl_expression.parse_cols(user_instruction)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        docs = task_instructions.df2text(self._obj, col_li)

        out = sem_schema(
            docs=docs,
            model=lotus.settings.lm,
            columns=columns,
            col_description=col_description,
            postprocessor=postprocessor,
        )

        new_df = pd.DataFrame()

        for column, value in zip(columns, out.outputs):
            new_df[column] = value

        new_df = new_df.reset_index(drop=True)

        return new_df
