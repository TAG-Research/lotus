from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

import lotus
from lotus.templates import task_instructions

from .postprocessors import filter_postprocess


def sem_filter(
    docs: List[str],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_df_txt: Optional[str] = None,
    examples_answers: Optional[List[bool]] = None,
    cot_reasoning: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    logprobs: bool = False,
) -> Tuple:
    """
    Filters a list of documents based on a given user instruction using a language model.

    Args:
        docs (List[str]): The list of documents to filter.
        model (lotus.models.LM): The language model used for filtering.
        user_instruction (str): The user instruction for filtering.
        default (Optional[bool]): The default value for filtering in case of parsing errors. Defaults to True.
        examples_df_txt (Optional[str]: The text for examples. Defaults to None.
        examples_answers (Optional[List[bool]]): The answers for examples. Defaults to None.
        cot_reasoning (Optional[List[str]]): The reasoning for CoT. Defaults to None.
        logprobs (Optional[bool]): Whether to return log probabilities. Defaults to False.

    Returns:
        Tuple: A tuple containing the True/False outputs, raw outputs, explanations, and raw log probabilities (if logprobs=True).
    """
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            doc, user_instruction, examples_df_txt, examples_answers, cot_reasoning, strategy
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)
    res = model(inputs, logprobs=logprobs)
    if logprobs:
        raw_outputs, raw_logprobs = res
    else:
        raw_outputs = res

    outputs, explanations = filter_postprocess(
        raw_outputs, default=default, cot_reasoning=strategy in ["cot", "zs-cot"]
    )
    lotus.logger.debug(f"outputs: {outputs}")
    lotus.logger.debug(f"raw_outputs: {raw_outputs}")
    lotus.logger.debug(f"explanations: {explanations}")

    if logprobs:
        return outputs, raw_outputs, explanations, raw_logprobs
    return outputs, raw_outputs, explanations


@pd.api.extensions.register_dataframe_accessor("sem_filter")
class SemFilterDataframe:
    """DataFrame accessor for semantic filter."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify that the Series has the correct type
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        user_instruction: str,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: Optional[pd.DataFrame] = None,
        helper_examples: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
        helper_strategy: Optional[str] = None,
        cascade_threshold: Optional[float] = None,
        return_stats: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Applies semantic filter over a dataframe.

        Args:
            user_instruction (str): The user instruction for filtering.
            return_raw_outputs (Optional[bool]): Whether to return raw outputs. Defaults to False.
            default (Optional[bool]): The default value for filtering in case of parsing errors. Defaults to True.
            suffix (Optional[str]): The suffix for the new columns. Defaults to "_filter".
            examples (Optional[pd.DataFrame]): The examples dataframe. Defaults to None.
            helper_examples (Optional[pd.DataFrame]): The helper examples dataframe. Defaults to None.
            strategy (Optional[str]): TThe reasoning strategy. Defaults to None.
            helper_strategy (Optional[str]): The reasoning strategy for helper. Defaults to None.
            cascade_threshold (Optional[float]): The threshold for cascading. Defaults to None.
            return_stats (Optional[bool]): Whether to return statistics. Defaults to False.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]: The filtered dataframe or a tuple containing the filtered dataframe and statistics.
        """
        stats = {}
        lotus.logger.debug(user_instruction)
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(col_li)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        df_txt = task_instructions.df2text(self._obj, col_li)
        lotus.logger.debug(df_txt)
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

        if cascade_threshold is not None:
            stats["filters_resolved_by_helper_model"] = 0
            stats["filters_resolved_by_large_model"] = 0

            # Get few-shot examples for small LM
            helper_examples_df_txt = None
            helper_examples_answers = None
            helper_cot_reasoning = None
            if helper_examples is not None:
                assert "Answer" in helper_examples.columns, "Answer must be a column in examples dataframe"
                helper_examples_df_txt = task_instructions.df2text(helper_examples, col_li)
                helper_examples_answers = helper_examples["Answer"].tolist()

                if helper_strategy == "cot":
                    helper_cot_reasoning = examples["Reasoning"].tolist()

            # Run small LM and get logits
            helper_outputs, helper_raw_outputs, helper_explanations, helper_logprobs = sem_filter(
                df_txt,
                lotus.settings.helper_lm,
                formatted_usr_instr,
                default=default,
                examples_df_txt=helper_examples_df_txt,
                examples_answers=helper_examples_answers,
                cot_reasoning=helper_cot_reasoning,
                logprobs=True,
                strategy=helper_strategy,
            )

            high_conf_idxs = set()
            helper_tokens, helper_confidences = lotus.settings.helper_lm.format_logprobs_for_cascade(helper_logprobs)

            # Find where true/false is said and look at confidence
            for idx_i, (tokens, confidences) in enumerate(zip(helper_tokens, helper_confidences)):
                for idx_j in range(len(tokens) - 1, -1, -1):
                    if tokens[idx_j].strip(" \n").lower() in ["true", "false"]:
                        conf = confidences[idx_j]
                        if conf >= cascade_threshold:
                            high_conf_idxs.add(idx_i)

            outputs, raw_outputs, explanations = (
                [None] * len(df_txt),
                [None] * len(df_txt),
                [None] * len(df_txt),
            )

            for idx in high_conf_idxs:
                outputs[idx] = helper_outputs[idx]
                raw_outputs[idx] = helper_raw_outputs[idx]
                explanations[idx] = helper_explanations[idx]

            # Send low confidence samples to large LM if any
            low_conf_idxs = sorted([i for i in range(len(helper_outputs)) if i not in high_conf_idxs])
            low_conf_df_txt = [df_txt[idx] for idx in low_conf_idxs]
            if low_conf_idxs:
                large_outputs, large_raw_outputs, large_explanations = sem_filter(
                    low_conf_df_txt,
                    lotus.settings.lm,
                    formatted_usr_instr,
                    default=default,
                    examples_df_txt=examples_df_txt,
                    examples_answers=examples_answers,
                    cot_reasoning=cot_reasoning,
                    strategy=strategy,
                )

                for idx, large_idx in enumerate(low_conf_idxs):
                    outputs[large_idx] = large_outputs[idx]
                    raw_outputs[large_idx] = large_raw_outputs[idx]
                    explanations[large_idx] = large_explanations[idx]

            stats["filters_resolved_by_helper_model"] += len(high_conf_idxs)
            stats["filters_resolved_by_large_model"] += len(low_conf_idxs)

        else:
            outputs, raw_outputs, explanations = sem_filter(
                df_txt,
                lotus.settings.lm,
                formatted_usr_instr,
                default=default,
                examples_df_txt=examples_df_txt,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
            )

        # find indices where output is True
        ids = [i for i, x in enumerate(outputs) if x]
        idx_ids = [self._obj.index[i] for i, x in enumerate(outputs) if x]
        lotus.logger.debug(f"ids: {ids}")
        lotus.logger.debug(f"idx_ids: {idx_ids}")

        [outputs[i] for i in ids]
        filtered_explanations = [explanations[i] for i in ids]
        filtered_raw_outputs = [raw_outputs[i] for i in ids]
        lotus.logger.debug(f"filtered_raw_outputs: {filtered_raw_outputs}")

        new_df = self._obj.iloc[ids]
        new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)

        # return rows where output is True
        if return_explanations and return_raw_outputs:
            new_df["explanation" + suffix] = filtered_explanations
            new_df["raw_output" + suffix] = filtered_raw_outputs
        elif return_explanations:
            new_df["explanation" + suffix] = filtered_explanations
        elif return_raw_outputs:
            new_df["raw_output" + suffix] = filtered_raw_outputs

        if return_stats:
            return new_df, stats

        return new_df
