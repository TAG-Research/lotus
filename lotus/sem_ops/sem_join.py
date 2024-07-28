from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import lotus

from .sem_filter import sem_filter


def sem_join(
    l1: pd.Series,
    l2: pd.Series,
    ids1: List[int],
    ids2: List[int],
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    examples_df_txt: Optional[str] = None,
    examples_answers: Optional[List[bool]] = None,
    cot_reasoning: Optional[List[str]] = None,
    default: bool = True,
    strategy: Optional[str] = None,
) -> Tuple:
    """
    Joins two series using a model.

    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        ids1 (List[int]): The ids for the first series.
        ids2 (List[int]): The ids for the second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        model (lotus.models.LM): The model to use.
        user_instruction (str): The user instruction for join.
        examples_df_txt (Optional[str]): The examples dataframe text. Defaults to None.
        examples_answers (Optional[List[bool]]): The answers for examples. Defaults to None.
        cot_reasoning (Optional[List[str]]): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.

    Returns:
        Tuple: The join results, filter outputs, all raw outputs, and all explanations.
    """
    filter_outputs = []
    all_raw_outputs = []
    all_explanations = []

    join_results = []

    # for i1 in enumerate(l1):
    for id1, i1 in zip(ids1, l1):
        # perform llm filter
        modified_docs = l2.apply(lambda doc: f"{col1_label}: {i1}\n{col2_label}: {doc}")
        outputs, raw_outputs, explanations = sem_filter(
            modified_docs,
            model,
            user_instruction,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            default=default,
            strategy=strategy,
        )
        filter_outputs.extend(outputs)
        all_raw_outputs.extend(raw_outputs)
        all_explanations.extend(explanations)

        join_results.extend(
            [
                (id1, ids2[i], explanation)
                for i, (output, explanation) in enumerate(zip(outputs, explanations))
                if output
            ]
        )

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    return join_results, filter_outputs, all_raw_outputs, all_explanations


def sem_join_cascade(
    l1: pd.Series,
    l2: pd.Series,
    ids1: List[int],
    ids2: List[int],
    col1_label: str,
    col2_label: str,
    user_instruction: str,
    cascade_threshold: int,
    examples_df_txt: Optional[str] = None,
    examples_answers: Optional[List[bool]] = None,
    cot_reasoning: Optional[List[str]] = None,
    default: bool = True,
    strategy: Optional[str] = None,
) -> List[str]:
    filter_outputs = []
    all_raw_outputs = []
    all_explanations = []

    join_results = []
    num_helper = 0
    num_large = 0

    for id1, i1 in zip(ids1, l1):
        # perform llm filter
        modified_docs = l2.apply(lambda doc: f"{col1_label}: {i1}\n{col2_label}: {doc}")
        helper_outputs, helper_raw_outputs, helper_explanations, helper_logprobs = sem_filter(
            modified_docs,
            lotus.settings.helper_lm,
            user_instruction,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            default=default,
            logprobs=True,
            strategy=strategy,
        )

        high_conf_idxs = set()
        for idx_i in range(len(helper_outputs)):
            # Get the logprobs
            if lotus.settings.helper_lm.provider == "vllm":
                tokens = helper_logprobs[idx_i]["tokens"]
                confidences = np.exp(helper_logprobs[idx_i]["token_logprobs"])
            elif lotus.settings.helper_lm.provider == "openai":
                content = helper_logprobs[idx_i]["content"]
                tokens = [content[t_idx]["token"] for t_idx in range(len(content))]
                confidences = np.exp([content[t_idx]["logprob"] for t_idx in range(len(content))])

            # Find where true/false is said and look at confidence
            for idx_j in range(len(tokens) - 1, -1, -1):
                if tokens[idx_j].strip(" \n").lower() in ["true", "false"]:
                    conf = confidences[idx_j]
                    if conf >= cascade_threshold:
                        high_conf_idxs.add(idx_i)

        # Send low confidence samples to large LM
        low_conf_idxs = sorted([i for i in range(len(helper_outputs)) if i not in high_conf_idxs])
        low_conf_docs = [modified_docs[idx] for idx in low_conf_idxs]

        large_outputs, large_raw_outputs, large_explanations = sem_filter(
            low_conf_docs,
            lotus.settings.lm,
            user_instruction,
            default=default,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
        )

        outputs, raw_outputs, explanations = (
            [None] * len(modified_docs),
            [None] * len(modified_docs),
            [None] * len(modified_docs),
        )
        for idx in high_conf_idxs:
            outputs[idx] = helper_outputs[idx]
            raw_outputs[idx] = helper_raw_outputs[idx]
            explanations[idx] = helper_explanations[idx]

        for idx, large_idx in enumerate(low_conf_idxs):
            outputs[large_idx] = large_outputs[idx]
            raw_outputs[large_idx] = large_raw_outputs[idx]
            explanations[large_idx] = large_explanations[idx]

        filter_outputs.extend(outputs)
        all_raw_outputs.extend(raw_outputs)
        all_explanations.extend(explanations)

        join_results.extend(
            [
                (id1, ids2[i], explanation)
                for i, (output, explanation) in enumerate(zip(outputs, explanations))
                if output
            ]
        )

        num_helper += len(high_conf_idxs)
        num_large += len(low_conf_idxs)

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    return (
        join_results,
        filter_outputs,
        all_raw_outputs,
        all_explanations,
        num_helper,
        num_large,
    )


@pd.api.extensions.register_dataframe_accessor("sem_join")
class SemJoinDataframe:
    """DataFrame accessor for semantic join."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        other: Union[pd.DataFrame, pd.Series],
        join_instruction: str,
        return_explanations: bool = False,
        how: str = "inner",
        suffix: str = "_join",
        examples: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
        default: bool = True,
        cascade_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Applies semantic join over a dataframe.

        Args:
            other (Union[pd.DataFrame, pd.Series]): The other dataframe or series to join with.
            join_instruction (str): The user instruction for join.
            return_explanations (bool): Whether to return explanations. Defaults to False.
            how (str): The type of join to perform. Defaults to "inner".
            suffix (str): The suffix for the new columns. Defaults to "_join".
            examples (Optional[pd.DataFrame]): The examples dataframe. Defaults to None.
            strategy (Optional[str]): The reasoning strategy. Defaults to None.
            default (bool): The default value for the join in case of parsing errors. Defaults to True.
            cascade_threshold (Optional[float]): The threshold for cascading. Defaults to None.

        Returns:
            pd.DataFrame: The dataframe with the new joined columns.
        """

        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        if how != "inner":
            raise NotImplementedError("Only inner join is currently supported")

        cols = lotus.nl_expression.parse_cols(join_instruction)
        left_on = None
        right_on = None
        for col in cols:
            if ":left" in col:
                left_on = col
                real_left_on = col.split(":left")[0]
            elif ":right" in col:
                right_on = col
                real_right_on = col.split(":right")[0]

        if left_on is None:
            for col in cols:
                if col in self._obj.columns:
                    left_on = col
                    real_left_on = col

                    if col in other.columns:
                        raise ValueError("Column found in both dataframes")
                    break
        if right_on is None:
            for col in cols:
                if col in other.columns:
                    right_on = col
                    real_right_on = col

                    if col in self._obj.columns:
                        raise ValueError("Column found in both dataframes")
                    break

        assert left_on is not None, "Column not found in left dataframe"
        assert right_on is not None, "Column not found in right dataframe"

        examples_df_txt = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_df_txt = []
            for idx, row in examples.iterrows():
                examples_df_txt.append(f"{left_on}: {row[real_left_on]}\n{right_on}: {row[real_right_on]}")
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot":
                return_explanations = True
                cot_reasoning = examples["Reasoning"].tolist()

        if cascade_threshold is not None:
            (
                join_results,
                filter_outputs,
                all_raw_outputs,
                all_explanations,
                num_helper,
                num_large,
            ) = sem_join_cascade(
                self._obj[real_left_on],
                other[real_right_on],
                self._obj.index,
                other.index,
                left_on,
                right_on,
                join_instruction,
                cascade_threshold,
                examples_df_txt=examples_df_txt,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                default=default,
                strategy=strategy,
            )
        else:
            join_results, filter_outputs, all_raw_outputs, all_explanations = sem_join(
                self._obj[real_left_on],
                other[real_right_on],
                self._obj.index,
                other.index,
                left_on,
                right_on,
                lotus.settings.lm,
                join_instruction,
                examples_df_txt=examples_df_txt,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                default=default,
                strategy=strategy,
            )
        lotus.logger.debug(f"join_results: {join_results}")
        lotus.logger.debug(f"all_raw_outputs: {all_raw_outputs}")

        df1 = self._obj.copy()
        df2 = other.copy()
        df1["_left_id"] = self._obj.index
        df2["_right_id"] = other.index
        # add suffix to column names
        for col in df1.columns:
            if col in df2.columns:
                df1.rename(columns={col: col + ":left"}, inplace=True)
                df2.rename(columns={col: col + ":right"}, inplace=True)

        if return_explanations:
            temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id", f"explanation{suffix}"])
        else:
            join_results = [(jr[0], jr[1]) for jr in join_results]
            temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id"])

        joined_df = (
            df1.join(temp_df.set_index("_left_id"), how="right", on="_left_id")
            .join(df2.set_index("_right_id"), how="left", on="_right_id")
            .drop(columns=["_left_id", "_right_id"])
        )

        if cascade_threshold is not None:
            return joined_df, num_helper, num_large

        return joined_df
