from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import lotus
from lotus.templates import task_instructions
from lotus.types import LMOutput, LogprobsForFilterCascade, SemanticFilterOutput
from lotus.utils import show_safe_mode

from .cascade_utils import calibrate_llm_logprobs, importance_sampling, learn_cascade_thresholds
from .postprocessors import filter_postprocess


def sem_filter(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
) -> SemanticFilterOutput:
    """
    Filters a list of documents based on a given user instruction using a language model.

    Args:
        docs (list[dict[str, Any]]): The list of documents to filter. Each document is a tuple of text and images.
        model (lotus.models.LM): The language model used for filtering.
        user_instruction (str): The user instruction for filtering.
        default (bool): The default value for filtering in case of parsing errors. Defaults to True.
        examples_multimodal_data (list[dict[str, Any]] | None): The text for examples. Defaults to None.
        examples_answers (list[bool] | None): The answers for examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.
        logprobs (bool): Whether to return log probabilities. Defaults to False.

    Returns:
        SemanticFilterOutput: The True/False outputs, raw outputs, and explanations, and log probabilities.
    """
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            doc, user_instruction, examples_multimodal_data, examples_answers, cot_reasoning, strategy
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)
    kwargs: dict[str, Any] = {"logprobs": logprobs}

    if safe_mode:
        estimated_total_calls = len(docs)
        estimated_total_cost = sum(model.count_tokens(input) for input in inputs)
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    lm_output: LMOutput = model(inputs, show_progress_bar=show_progress_bar, **kwargs)

    postprocess_output = filter_postprocess(
        lm_output.outputs, default=default, cot_reasoning=strategy in ["cot", "zs-cot"]
    )
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"raw_outputs: {postprocess_output.raw_outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")

    if safe_mode:
        model.print_total_usage()

    return SemanticFilterOutput(**postprocess_output.model_dump(), logprobs=lm_output.logprobs if logprobs else None)


def learn_filter_cascade_thresholds(
    sample_multimodal_data: list[dict[str, Any]],
    lm: lotus.models.LM,
    formatted_usr_instr: str,
    default: bool,
    recall_target: float,
    precision_target: float,
    delta: float,
    helper_true_probs: list[float],
    sample_correction_factors: NDArray[np.float64],
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> tuple[float, float]:
    """Automatically learns the cascade thresholds for a cascade
    filter given a sample of data and doing a search across threshold
    to see what threshold gives the best accuracy."""

    try:
        large_outputs = sem_filter(
            sample_multimodal_data,
            lm,
            formatted_usr_instr,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            safe_mode=False,
        ).outputs

        best_combination, _ = learn_cascade_thresholds(
            proxy_scores=helper_true_probs,
            oracle_outputs=large_outputs,
            sample_correction_factors=sample_correction_factors,
            recall_target=recall_target,
            precision_target=precision_target,
            delta=delta,
        )

        lotus.logger.info(f"Learned cascade thresholds: {best_combination}")
        return best_combination

    except Exception as e:
        lotus.logger.error(f"Error while learning filter cascade thresholds: {e}")
        raise e


@pd.api.extensions.register_dataframe_accessor("sem_filter")
class SemFilterDataframe:
    """DataFrame accessor for semantic filter."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
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
        examples: pd.DataFrame | None = None,
        helper_examples: pd.DataFrame | None = None,
        strategy: str | None = None,
        helper_strategy: str | None = None,
        learn_cascade_threshold_sample_percentage: int | None = None,
        recall_target: float | None = None,
        precision_target: float | None = None,
        failure_probability: float | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """
        Applies semantic filter over a dataframe.

        Args:
            user_instruction (str): The user instruction for filtering.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.
            default (bool): The default value for filtering in case of parsing errors. Defaults to True.
            suffix (str): The suffix for the new columns. Defaults to "_filter".
            examples (pd.DataFrame | None): The examples dataframe. Defaults to None.
            helper_examples (pd.DataFrame | None): The helper examples dataframe. Defaults to None.
            strategy (str | None): The reasoning strategy. Defaults to None.
            helper_strategy (str | None): The reasoning strategy for helper. Defaults to None.
            learn_cascade_threshold_sample_size (Optional[int]): The percentage of samples from which to learn thresholds when cascading.
            recall_target (float | None): The specified recall target when cascading.
            precision_target (float | None): The specified precision target when cascading.
            failure_probability (float | None): The specified failure probability for precision/recall targets when cascading.
            return_stats (bool): Whether to return statistics. Defaults to False.

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The filtered dataframe or a tuple containing the filtered dataframe and statistics.
        """
        stats = {}
        lotus.logger.debug(user_instruction)
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(col_li)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        lotus.logger.debug(multimodal_data)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot":
                return_explanations = True
                cot_reasoning = examples["Reasoning"].tolist()

        pos_cascade_threshold, neg_cascade_threshold = None, None
        if learn_cascade_threshold_sample_percentage is not None:
            # Get few-shot examples for small LM
            helper_examples_multimodal_data = None
            helper_examples_answers = None
            helper_cot_reasoning = None
            if helper_examples is not None:
                assert "Answer" in helper_examples.columns, "Answer must be a column in examples dataframe"
                helper_examples_multimodal_data = task_instructions.df2multimodal_info(helper_examples, col_li)
                helper_examples_answers = helper_examples["Answer"].tolist()

                if helper_strategy == "cot":
                    helper_cot_reasoning = examples["Reasoning"].tolist()

        if learn_cascade_threshold_sample_percentage and lotus.settings.helper_lm:
            if helper_strategy == "cot":
                lotus.logger.error("CoT not supported for helper models in cascades.")
                raise Exception

            if recall_target is None or precision_target is None or failure_probability is None:
                lotus.logger.error(
                    "Recall target, precision target, and confidence need to be specified for learned thresholds."
                )
                raise Exception

            # Run small LM and get logits
            helper_output = sem_filter(
                multimodal_data,
                lotus.settings.helper_lm,
                formatted_usr_instr,
                default=default,
                examples_multimodal_data=helper_examples_multimodal_data,
                examples_answers=helper_examples_answers,
                cot_reasoning=helper_cot_reasoning,
                logprobs=True,
                strategy=helper_strategy,
                safe_mode=safe_mode,
                show_progress_bar=True,
            )
            helper_outputs, helper_logprobs = helper_output.outputs, helper_output.logprobs
            formatted_helper_logprobs: LogprobsForFilterCascade = (
                lotus.settings.helper_lm.format_logprobs_for_filter_cascade(helper_logprobs)
            )
            helper_true_probs = calibrate_llm_logprobs(formatted_helper_logprobs.true_probs)

            sample_indices, correction_factors = importance_sampling(
                helper_true_probs, learn_cascade_threshold_sample_percentage
            )
            sample_df = self._obj.loc[sample_indices]
            sample_multimodal_data = task_instructions.df2multimodal_info(sample_df, col_li)
            sample_helper_true_probs = [helper_true_probs[i] for i in sample_indices]
            sample_correction_factors = correction_factors[sample_indices]

            pos_cascade_threshold, neg_cascade_threshold = learn_filter_cascade_thresholds(
                sample_multimodal_data=sample_multimodal_data,
                lm=lotus.settings.lm,
                formatted_usr_instr=formatted_usr_instr,
                default=default,
                recall_target=recall_target,
                precision_target=precision_target,
                delta=failure_probability / 2,
                helper_true_probs=sample_helper_true_probs,
                sample_correction_factors=sample_correction_factors,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
            )

            stats["pos_cascade_threshold"] = pos_cascade_threshold
            stats["neg_cascade_threshold"] = neg_cascade_threshold

        if pos_cascade_threshold is not None and neg_cascade_threshold is not None:
            stats["filters_resolved_by_helper_model"] = 0
            stats["filters_resolved_by_large_model"] = 0

            high_conf_idxs = set()

            # Find where true/false is said and look at confidence
            for idx_i in range(len(helper_true_probs)):
                true_prob = helper_true_probs[idx_i]
                if true_prob >= pos_cascade_threshold or true_prob <= neg_cascade_threshold:
                    high_conf_idxs.add(idx_i)
                    helper_outputs[idx_i] = (
                        True
                        if true_prob >= pos_cascade_threshold
                        else False
                        if true_prob <= neg_cascade_threshold
                        else helper_outputs[idx_i]
                    )

            lotus.logger.info(f"Num routed to smaller model: {len(high_conf_idxs)}")
            stats["num_routed_to_helper_model"] = len(high_conf_idxs)

            outputs: list[bool] = [False] * len(multimodal_data)
            raw_outputs: list[str] = [""] * len(multimodal_data)
            explanations: list[str | None] = [None] * len(multimodal_data)

            assert all(isinstance(x, str) for x in helper_output.explanations) or all(
                x is None for x in helper_output.explanations
            )
            for idx in high_conf_idxs:
                outputs[idx] = helper_outputs[idx]
                raw_outputs[idx] = helper_output.raw_outputs[idx]
                explanations[idx] = helper_output.explanations[idx]

            # Send low confidence samples to large LM if any
            low_conf_idxs = sorted([i for i in range(len(helper_outputs)) if i not in high_conf_idxs])
            low_conf_multimodal_data = [multimodal_data[idx] for idx in low_conf_idxs]
            if low_conf_idxs:
                large_output = sem_filter(
                    low_conf_multimodal_data,
                    lotus.settings.lm,
                    formatted_usr_instr,
                    default=default,
                    examples_multimodal_data=examples_multimodal_data,
                    examples_answers=examples_answers,
                    cot_reasoning=cot_reasoning,
                    strategy=strategy,
                    safe_mode=safe_mode,
                )

                for idx, large_idx in enumerate(low_conf_idxs):
                    outputs[large_idx] = large_output.outputs[idx]
                    raw_outputs[large_idx] = large_output.raw_outputs[idx]
                    explanations[large_idx] = large_output.explanations[idx]

            stats["filters_resolved_by_helper_model"] += len(high_conf_idxs)
            stats["filters_resolved_by_large_model"] += len(low_conf_idxs)

        else:
            output = sem_filter(
                multimodal_data,
                lotus.settings.lm,
                formatted_usr_instr,
                default=default,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
                safe_mode=safe_mode,
                show_progress_bar=True,
            )
            outputs = output.outputs
            raw_outputs = output.raw_outputs
            explanations = output.explanations

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
