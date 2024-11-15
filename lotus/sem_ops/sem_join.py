from typing import Any

import pandas as pd

import lotus
from lotus.types import SemanticJoinOutput
from lotus.templates import task_instructions

from .sem_filter import sem_filter
from .cascade_utils import importance_sampling, learn_cascade_thresholds, calibrate_sem_sim_join


def sem_join(
    l1: pd.Series,
    l2: pd.Series,
    ids1: list[int],
    ids2: list[int],
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    examples_df_txt: list[str] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: str | None = None,
) -> SemanticJoinOutput:
    """
    Joins two series using a model.

    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        ids1 (list[int]): The ids for the first series.
        ids2 (list[int]): The ids for the second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        model (lotus.models.LM): The model to use.
        user_instruction (str): The user instruction for join.
        examples_df_txt (list[str] | None): The examples dataframe text. Defaults to None.
        examples_answers (list[bool] | None): The answers for examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.

    Returns:
        SemanticJoinOutput: The join results, filter outputs, all raw outputs, and all explanations.
    """
    filter_outputs = []
    all_raw_outputs = []
    all_explanations = []

    join_results = []

    # for i1 in enumerate(l1):
    for id1, i1 in zip(ids1, l1):
        # perform llm filter
        modified_docs = l2.apply(lambda doc: f"{col1_label}: {i1}\n{col2_label}: {doc}")
        output = sem_filter(
            modified_docs,
            model,
            user_instruction,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            default=default,
            strategy=strategy,
        )
        outputs = output.outputs
        raw_outputs = output.raw_outputs
        explanations = output.explanations

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

    return SemanticJoinOutput(
        join_results=join_results,
        filter_outputs=filter_outputs,
        all_raw_outputs=all_raw_outputs,
        all_explanations=all_explanations,
    )


def sem_join_cascade(
    l1: pd.Series,
    l2: pd.Series,
    ids1: list[int],
    ids2: list[int],
    col1_label: str,
    col2_label: str,
    user_instruction: str,
    recall_target: float,
    precision_target: float,
    sampling_percentage: float = 0.1,
    failure_probability: float = 0.2,
    examples_df_txt: str | None = None,
    examples_answers: list[bool] | None = None,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: str | None = None,
    sampling_range: tuple[int, int] | None = None,
) -> SemanticJoinOutput:
    """
    Joins two series using a cascade helper model and a large model.
    
    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        ids1 (list[int]): The ids for the first series.
        ids2 (list[int]): The ids for the second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        user_instruction (str): The user instruction for join.
        recall_target (float): The target recall.
        precision_target (float): The target precision.
        sampling_percentage (float): The percentage of the data to sample. Defaults to 0.1.
        failure_probability (float): The failure probability. Defaults to 0.2.
        examples_df_txt (Optional[str]): The examples dataframe text. Defaults to None.
        examples_answers (Optional[list[bool]]): The answers for examples. Defaults to None.
        map_instruction (Optional[str]): The map instruction. Defaults to None.
        map_examples (Optional[pd.DataFrame]): The map examples. Defaults to None.
        cot_reasoning (Optional[list[str]]): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        strategy (Optional[str]): The reasoning strategy. Defaults to None.
        sampling_range (Optional[tuple[int, int]]): The sampling range. Defaults to None.
        
    Returns:
        SemanticJoinOutput: The join results, filter outputs, all raw outputs, and all explanations.
        
        Note that filter_outputs, all_raw_outputs, and all_explanations are empty list because
        the helper model do not generate these outputs.
    """
    filter_outputs: list[bool] = []
    all_raw_outputs: list[str] = []
    all_explanations: list[str | None] = []

    join_results: list[tuple[int, int, str | None]] = []
    num_helper = 0
    num_large = 0

    # Determine the join plan
    helper_high_conf, helper_low_conf = join_optimizer(
        recall_target,
        precision_target,
        l1,
        l2,
        col1_label,
        col2_label,
        user_instruction,
        sampling_percentage=sampling_percentage,
        failure_probability=failure_probability,
        examples_df_txt=examples_df_txt,
        examples_answers=examples_answers,
        map_instruction=map_instruction,
        map_examples=map_examples,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
        sampling_range=sampling_range,
        )

    num_helper = len(helper_high_conf)
    num_large = len(helper_low_conf)
    
    # Accept helper results with high confidence
    print(f"helper_high_conf = {helper_high_conf}")
    join_results = [(row['_left_id'], row['_right_id'], None) for _, row in helper_high_conf.iterrows()]

    # Send low confidence rows to large LM
    for unique_l1 in helper_low_conf[col1_label].unique():
        unique_l1_id = helper_low_conf[helper_low_conf[col1_label] == unique_l1]['_left_id'].iloc[0]
        l2_for_l1 = helper_low_conf[helper_low_conf[col1_label] == unique_l1][col2_label]
        l2_for_l1_index = helper_low_conf[helper_low_conf[col1_label] == unique_l1]['_right_id']
        large_join_output = sem_join(
            pd.Series([unique_l1]),
            l2_for_l1,
            [unique_l1_id],
            l2_for_l1_index.tolist(),
            col1_label,
            col2_label,
            lotus.settings.lm,
            user_instruction,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            default=default,
            strategy=strategy,
        )
    
        join_results.extend(large_join_output.join_results)

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    stats = {"filters_resolved_by_helper_model": num_helper, "filters_resolved_by_large_model": num_large}
    return SemanticJoinOutput(
        join_results=join_results,
        filter_outputs=filter_outputs,
        all_raw_outputs=all_raw_outputs,
        all_explanations=all_explanations,
        stats=stats,
    )


def run_sem_sim_join(
    l1: pd.Series,
    l2: pd.Series,
    col1_label: str,
    col2_label: str
) -> pd.DataFrame:
    """
    Wrapper function to run sem_sim_join in sem_join then calibrate the scores for approximate join
    
    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        
    Returns:
        pd.DataFrame: The similarity join results.
    """
    # Transform the series into DataFrame
    if isinstance(l1, pd.Series):
        l1_df = l1.to_frame(name=col1_label)
    elif isinstance(l1, pd.DataFrame):
        l1_df = l1
    else:
        lotus.logger.error("l1 must be a pandas Series or DataFrame")

    l2_df = l2.to_frame(name=col2_label)
    l2_df = l2_df.sem_index(col2_label, f"{col2_label}_index")

    K = len(l2) * len(l1)
    # Run sem_sim_join as helper on the sampled data
    out = l1_df.sem_sim_join(
        l2_df, 
        left_on=col1_label, 
        right_on=col2_label, 
        K=K, 
        keep_index=True)
    
    out['_scores'].to_csv("raw_helper_join.csv")

    # Correct helper scores
    out['_scores'] = calibrate_sem_sim_join(out['_scores'].tolist())
    return out


def map_l1_to_l2(
    l1: pd.Series,
    col1_label: str,
    col2_label: str,
    map_instruction: str = None,
    map_examples: pd.DataFrame = None
) -> tuple[pd.DataFrame, str]:
    """
    Wrapper function to run sem_map in sem_join.
    
    Args:
        l1 (pd.Series): The first series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        map_instruction (str): The map instruction. Defaults to None.
        map_examples (pd.DataFrame): The map examples. Defaults to None.
        
    Returns:
        tuple[pd.DataFrame, str]: The mapped DataFrame and the mapped column name.
    """
    if ":left" in col1_label:
        real_left_on = col1_label.split(":left")[0]
    else:
        real_left_on = col1_label
        
    if ":right" in col2_label:
        real_right_on = col2_label.split(":right")[0]
    else:
        real_right_on = col2_label
    
    inst = ""
    if map_instruction:
        inst = map_instruction
    else:
        default_map_instruction = f"Given {{{real_left_on}}}, identify the most relevant {real_right_on}. Always write your answer as a list of 2-10 comma-separated {real_right_on}."
        inst = default_map_instruction

    # Transform l1 into DataFrame for sem_map
    l1_df = l1.to_frame(name=real_left_on)
    mapped_col1_name = f"_{col1_label}"
    
    # Map l1 to l2
    out = l1_df.sem_map(
        inst,
        suffix=mapped_col1_name,
        examples=map_examples)
    out = out.rename(columns={real_left_on: col1_label})

    return out, mapped_col1_name


def join_optimizer(
    recall_target: float,
    precision_target: float,
    l1: pd.Series,
    l2: pd.Series,
    col1_label: str,
    col2_label: str,
    user_instruction: str,
    sampling_percentage: float = 0.1,
    failure_probability: float = 0.2,
    examples_df_txt: str | None = None,
    examples_answers: list[bool] | None = None,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: str | None = None,
    sampling_range: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find most cost-effective join plan between Search-Filter and Map-Search-Filter 
    while satisfying the recall and precision target.
    
    Args:
        recall_target (float): The target recall.
        precision_target (float): The target precision.
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        user_instruction (str): The user instruction for join.
        sampling_percentage (float): The percentage of the data to sample. Defaults to 0.1.
        failure_probability (float): The failure probability. Defaults to 0.2.
        examples_df_txt (Optional[str]): The examples dataframe text. Defaults to None.
        examples_answers (Optional[list[bool]]): The answers for examples. Defaults to None.
        map_instruction (Optional[str]): The map instruction. Defaults to None.
        map_examples (Optional[pd.DataFrame]): The map examples. Defaults to None.
        cot_reasoning (Optional[list[str]]): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        strategy (Optional[str]): The reasoning strategy. Defaults to None.
        sampling_range (Optional[tuple[int, int]]): The sampling range. Defaults to None.
    
    returns:
        tuple[pd.DataFrame, pd.DataFrame]: The high confidence and low confidence join results.
    """
    
    # Helper is currently default to similiarity join
    if lotus.settings.helper_lm is not None:
        lotus.logger.debug("Helper model is not supported yet. Default to similarity join.")

    # Learn search-filter thresholds
    sf_helper_join = run_sem_sim_join(l1, l2, col1_label, col2_label)
    sf_t_pos, sf_t_neg, _ = learn_join_cascade_threshold(
        sf_helper_join, 
        recall_target, 
        precision_target, 
        col1_label,
        col2_label, 
        user_instruction, 
        sampling_percentage,
        delta=failure_probability / 2,
        examples_df_txt=examples_df_txt,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
        sampling_range=sampling_range)
    sf_high_conf = sf_helper_join[sf_helper_join['_scores'] >= sf_t_pos]
    sf_low_conf = sf_helper_join[(sf_helper_join['_scores'] < sf_t_pos) & (sf_helper_join['_scores'] > sf_t_neg)]
    sf_cost = len(sf_low_conf)

    # Learn map-search-filter thresholds
    mapped_l1, mapped_col1_label = map_l1_to_l2(l1, col1_label, col2_label, map_instruction=map_instruction, map_examples=map_examples)
    msf_helper_join = run_sem_sim_join(mapped_l1, l2, mapped_col1_label, col2_label)
    msf_t_pos, msf_t_neg, _ = learn_join_cascade_threshold(
        msf_helper_join, 
        recall_target, 
        precision_target, 
        col1_label,
        col2_label,
        user_instruction, 
        sampling_percentage,
        delta=failure_probability / 2,
        examples_df_txt=examples_df_txt,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        strategy=strategy,
        sampling_range=sampling_range)
    msf_high_conf = msf_helper_join[msf_helper_join['_scores'] >= msf_t_pos]
    msf_low_conf = msf_helper_join[(msf_helper_join['_scores'] < msf_t_pos) & (msf_helper_join['_scores'] > msf_t_neg)]
    msf_cost = len(msf_low_conf)

    # Select the cheaper join plan
    lotus.logger.info(f"Join Optimizer: plan cost analysis:")
    lotus.logger.info(f"    Search-Filter: {sf_cost} LLM calls, {len(sf_high_conf)} helper results.")
    lotus.logger.info(f"    Map-Search-Filter: {msf_cost} LLM calls, {len(msf_high_conf)} helper results.")

    if sf_cost < msf_cost:
        lotus.logger.info("Proceeding with Search-Filter")
        sf_high_conf = sf_high_conf.sort_values(by='_scores', ascending=False)
        sf_low_conf = sf_low_conf.sort_values(by='_scores', ascending=False)
        return sf_high_conf, sf_low_conf
    else:
        lotus.logger.info("Proceeding with Map-Search-Filter")
        msf_high_conf = msf_high_conf.sort_values(by='_scores', ascending=False)
        msf_low_conf = msf_low_conf.sort_values(by='_scores', ascending=False)
        return msf_high_conf, msf_low_conf


def learn_join_cascade_threshold(
    helper_join: pd.DataFrame,
    recall_target: float,
    precision_target: float,
    col1_label: str,
    col2_label: str,
    user_instruction: str,
    sampling_percentage: float = 0.1,
    delta: float = 0.2,
    examples_df_txt: str | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    strategy: str | None = None,
    sampling_range: tuple[int, int] | None = None,
) -> tuple[float, float, int]:
    """
    Extract a small sample of the data and find the optimal threshold pair that satisfies the recall and 
    precision target.

    Args:
        model (lotus.models.LM): The model to use.
        helper_join (pd.DataFrame): The helper join results.
        user_instruction (str): The user instruction for join.
        recall_target (float): The target recall.
        precision_target (float): The target precision.
        examples_df_txt (Optional[str]): The examples dataframe text. Defaults to None.
        examples_answers (Optional[list[bool]]): The answers for examples. Defaults to None.
        cot_reasoning (Optional[list[str]]): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        strategy (Optional[str]): The reasoning strategy. Defaults to None.
        sampling_percentage (float): The percentage of the data to sample. Defaults to 0.1.
    Returns:
        tuple: The join results, filter outputs, all raw outputs, and all explanations.
    """
    # Sample a small subset of the helper join result
    helper_scores = helper_join['_scores'].tolist()
    
    sample_indices, correction_factors = importance_sampling(helper_scores, sampling_percentage, sampling_range=sampling_range)

    sample_df = helper_join.iloc[sample_indices]
    sample_scores = sample_df['_scores'].tolist()
    sample_correction_factors = correction_factors[sample_indices]

    col_li = [col1_label, col2_label]
    sample_df_txt = task_instructions.df2text(sample_df, col_li)

    try:
        output = sem_filter(
            sample_df_txt,
            lotus.settings.lm,
            user_instruction,
            default=default,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
        )

        (pos_threshold, neg_threshold), large_calls = learn_cascade_thresholds(
            proxy_scores=sample_scores,
            oracle_outputs=output.outputs,
            sample_correction_factors=sample_correction_factors,
            recall_target=recall_target,
            precision_target=precision_target,
            delta=delta
        )

        lotus.logger.info(f"Learned cascade thresholds: {(pos_threshold, neg_threshold)}")

    except Exception as e:
        lotus.logger.error(f"Error while learning filter cascade thresholds: {e}")
        lotus.logger.error(f"Default to full join.")
        return 1, 0, float('inf')
    
    return pos_threshold, neg_threshold, large_calls


@pd.api.extensions.register_dataframe_accessor("sem_join")
class SemJoinDataframe:
    """DataFrame accessor for semantic join."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        other: pd.DataFrame | pd.Series,
        join_instruction: str,
        return_explanations: bool = False,
        how: str = "inner",
        suffix: str = "_join",
        examples: pd.DataFrame | None = None,
        strategy: str | None = None,
        default: bool = True,
        recall_target: float | None = None,
        precision_target: float | None = None,
        sampling_percentage: float | None = 0.1,
        failure_probability: float | None = 0.2,
        map_instruction: str | None = None,
        map_examples: pd.DataFrame | None = None,
        sampling_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """
        Applies semantic join over a dataframe.

        Args:
            other (pd.DataFrame | pd.Series): The other dataframe or series to join with.
            join_instruction (str): The user instruction for join.
            return_explanations (bool): Whether to return explanations. Defaults to False.
            how (str): The type of join to perform. Defaults to "inner".
            suffix (str): The suffix for the new columns. Defaults to "_join".
            examples (pd.DataFrame | None): The examples dataframe. Defaults to None.
            strategy (str | None): The reasoning strategy. Defaults to None.
            default (bool): The default value for the join in case of parsing errors. Defaults to True.
            
        Cascade-specific Arguments:
            recall_target (float | None): The target recall. Defaults to None.
            precision_target (float | None): The target precision. Defaults to None.
            sampling_percentage (float): The percentage of the data to sample. Defaults to 0.1.
            failure_probability (float): The failure probability. Defaults to 0.2.
            map_instruction (str): The map instruction. Defaults to None.
            map_examples (pd.DataFrame): The map examples. Defaults to None.
            sampling_range (tuple[int, int]): The sampling range. Defaults to None.

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

        num_full_join = len(self._obj) * len(other)

        if (recall_target is not None or precision_target is not None) and \
            (num_full_join >= lotus.settings.min_join_cascade_size):
            recall_target = 1.0 if recall_target is None else recall_target
            precision_target = 1.0 if precision_target is None else precision_target
            output = sem_join_cascade(
                self._obj[real_left_on],
                other[real_right_on],
                self._obj.index,
                other.index,
                left_on,
                right_on,
                join_instruction,
                recall_target,
                precision_target,
                sampling_percentage,
                failure_probability,
                examples_df_txt=examples_df_txt,
                examples_answers=examples_answers,
                map_instruction=map_instruction,
                map_examples=map_examples,
                cot_reasoning=cot_reasoning,
                default=default,
                strategy=strategy,
                sampling_range=sampling_range,
            )
        else:
            output = sem_join(
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
        join_results = output.join_results
        all_raw_outputs = output.all_raw_outputs

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
            temp_df = pd.DataFrame([(jr[0], jr[1]) for jr in join_results], columns=["_left_id", "_right_id"])

        joined_df = (
            df1.join(temp_df.set_index("_left_id"), how="right", on="_left_id")
            .join(df2.set_index("_right_id"), how="left", on="_right_id")
            .drop(columns=["_left_id", "_right_id"])
        )

        if output.stats:
            return joined_df, output.stats

        return joined_df
