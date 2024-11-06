import heapq
import re
from typing import Any

import numpy as np
import pandas as pd

import lotus
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticTopKOutput


def get_match_prompt_binary(
    doc1: str, doc2: str, user_instruction: str, strategy: str | None = None
) -> list[dict[str, Any]]:
    if strategy == "zs-cot":
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'First give your reasoning. Then you MUST end your output with "Answer: Document 1 or Document 2"\n'
            'You must pick a number and cannot say things like "None" or "Neither"\n'
            'Remember to explicitly state "Answer:" at the end before your choice.'
        )
    else:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'Respond only with the label of the document such as "Document NUMBER".\n'
            "NUMBER must be either 1 or 2, depending on which document is most relevant.\n"
            'You must pick a number and cannot say things like "None" or "Neither"'
        )

    prompt = f"Question: {user_instruction}\n\n"
    for idx, doc in enumerate([doc1, doc2]):
        prompt = f"{prompt}\nDocument {idx+1}:\n{doc}\n"

    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({"role": "user", "content": prompt})
    return messages


def parse_ans_binary(answer: str) -> bool:
    lotus.logger.debug(f"Response from model: {answer}")
    try:
        matches = list(re.finditer(r"Document[\s*](\d+)", answer, re.IGNORECASE))
        if len(matches) == 0:
            matches = list(re.finditer(r"(\d+)", answer, re.IGNORECASE))
        ans = int(matches[-1].group(1)) - 1
        if ans not in [0, 1]:
            lotus.logger.info(f"Could not parse {answer}")
            return True
        return ans == 0
    except Exception:
        lotus.logger.info(f"Could not parse {answer}")
        return True


def compare_batch_binary(
    pairs: list[tuple[str, str]], user_instruction: str, strategy: str | None = None
) -> tuple[list[bool], int]:
    match_prompts = []
    tokens = 0
    for doc1, doc2 in pairs:
        match_prompts.append(get_match_prompt_binary(doc1, doc2, user_instruction, strategy=strategy))
        tokens += lotus.settings.lm.count_tokens(match_prompts[-1])
    lm_results: LMOutput = lotus.settings.lm(match_prompts)
    results: list[bool] = list(map(parse_ans_binary, lm_results.outputs))
    return results, tokens


def compare_batch_binary_cascade(
    pairs: list[tuple[str, str]],
    user_instruction: str,
    cascade_threshold: float,
    strategy: str | None = None,
) -> tuple[list[bool], int, int, int]:
    match_prompts = []
    small_tokens = 0
    for doc1, doc2 in pairs:
        match_prompts.append(get_match_prompt_binary(doc1, doc2, user_instruction, strategy=strategy))
        small_tokens += lotus.settings.helper_lm.count_tokens(match_prompts[-1])

    results, helper_logprobs = lotus.settings.helper_lm(match_prompts, logprobs=True)
    helper_tokens, helper_confidences = lotus.settings.helper_lm.format_logprobs_for_cascade(helper_logprobs)

    parsed_results = []
    high_conf_idxs = set()
    for idx, res in enumerate(results):
        parsed_res = parse_ans_binary(res)
        parsed_results.append(parsed_res)

        # Find where docunent number is said and look at confidence
        for idx_j in range(len(helper_tokens[idx]) - 1, -1, -1):
            if helper_tokens[idx][idx_j].strip(" \n").isnumeric():
                conf = helper_confidences[idx][idx_j]
                if conf >= cascade_threshold:
                    high_conf_idxs.add(idx)

    large_tokens = 0
    num_large_calls = 0
    if len(high_conf_idxs) != len(helper_logprobs):
        # Send low confidence samples to large LM
        low_conf_idxs = sorted([i for i in range(len(helper_logprobs)) if i not in high_conf_idxs])

        large_match_prompts = []
        for i in low_conf_idxs:
            large_match_prompts.append(match_prompts[i])
            large_tokens += lotus.settings.lm.count_tokens(large_match_prompts[-1])

        large_lm_results: LMOutput = lotus.settings.lm(large_match_prompts)
        for idx, res in enumerate(large_lm_results.outputs):
            new_idx = low_conf_idxs[idx]
            parsed_res = parse_ans_binary(res)
            parsed_results[new_idx] = parsed_res

        num_large_calls = len(low_conf_idxs)
    return parsed_results, small_tokens, large_tokens, num_large_calls


def llm_naive_sort(
    docs: list[str],
    user_instruction: str,
    strategy: str | None = None,
) -> SemanticTopKOutput:
    """
    Sorts the documents using a naive quadratic method.

    Args:
        docs (list[str]): The list of documents to sort.
        user_instruction (str): The user instruction for sorting.

    Returns:
        SemanticTopKOutput: The indexes of the top k documents and stats.
    """
    N = len(docs)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((docs[i], docs[j]))

    llm_calls = len(pairs)
    comparisons, tokens = compare_batch_binary(pairs, user_instruction, strategy=strategy)
    votes = [0] * N
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if comparisons[idx]:
                votes[i] += 1
            else:
                votes[j] += 1
            idx += 1

    indexes = sorted(range(len(votes)), key=lambda i: votes[i], reverse=True)

    stats = {"total_tokens": tokens, "total_llm_calls": llm_calls}
    return SemanticTopKOutput(indexes=indexes, stats=stats)


def llm_quicksort(
    docs: list[str],
    user_instruction: str,
    K: int,
    embedding: bool = False,
    strategy: str | None = None,
    cascade_threshold: float | None = None,
) -> SemanticTopKOutput:
    """
    Sorts the documents using quicksort.

    Args:
        docs (list[str]): The list of documents to sort.
        user_instruction (str): The user instruction for sorting.
        K (int): The number of documents to return.
        embedding (bool): Whether to use embedding optimization.
        cascade_threshold (float | None): The confidence threshold for cascading to a larger model.

    Returns:
        SemanticTopKOutput: The indexes of the top k documents and stats
    """
    stats = {}
    stats["total_tokens"] = 0
    stats["total_llm_calls"] = 0

    if cascade_threshold is not None:
        stats["total_small_tokens"] = 0
        stats["total_large_tokens"] = 0
        stats["total_small_calls"] = 0
        stats["total_large_calls"] = 0

    def partition(indexes: list[int], low: int, high: int, K: int) -> int:
        nonlocal stats
        i = low - 1

        if embedding:
            # With embedding optimization
            if K <= high - low:
                pivot_value = heapq.nsmallest(K, indexes[low : high + 1])[-1]
            else:
                pivot_value = heapq.nsmallest(int((high - low + 1) / 2), indexes[low : high + 1])[-1]
            pivot_index = indexes.index(pivot_value)
        else:
            # Without embedding optimization
            pivot_index = np.random.randint(low, high + 1)
            pivot_value = indexes[pivot_index]

        pivot = docs[pivot_value]
        indexes[pivot_index], indexes[high] = indexes[high], indexes[pivot_index]

        pairs = [(docs[indexes[j]], pivot) for j in range(low, high)]
        if cascade_threshold is None:
            comparisons, tokens = compare_batch_binary(pairs, user_instruction, strategy=strategy)
            stats["total_tokens"] += tokens
            stats["total_llm_calls"] += len(pairs)
        else:
            comparisons, small_tokens, large_tokens, num_large_calls = compare_batch_binary_cascade(
                pairs,
                user_instruction,
                cascade_threshold,
                strategy=strategy,
            )
            stats["total_small_tokens"] += small_tokens
            stats["total_large_tokens"] += large_tokens
            stats["total_small_calls"] += len(pairs)
            stats["total_large_calls"] += num_large_calls

        for j, doc1_is_better in enumerate(comparisons, start=low):
            if doc1_is_better:
                i += 1
                indexes[i], indexes[j] = indexes[j], indexes[i]

        indexes[i + 1], indexes[high] = indexes[high], indexes[i + 1]
        return i + 1

    def quicksort_recursive(indexes: list[int], low: int, high: int, K: int) -> None:
        if high <= low:
            return

        if low < high:
            pi = partition(indexes, low, high, K)
            left_size = pi - low
            if left_size + 1 >= K:
                quicksort_recursive(indexes, low, pi - 1, K)
            else:
                quicksort_recursive(indexes, low, pi - 1, left_size)
                quicksort_recursive(indexes, pi + 1, high, K - left_size - 1)

    indexes = list(range(len(docs)))
    quicksort_recursive(indexes, 0, len(indexes) - 1, K)

    return SemanticTopKOutput(indexes=indexes, stats=stats)


class HeapDoc:
    """Class to define a document for the heap. Keeps track of the number of calls and tokens."""

    num_calls: int = 0
    total_tokens: int = 0
    strategy: str | None = None

    def __init__(self, doc: str, user_instruction: str, idx: int) -> None:
        self.doc = doc
        self.user_instruction = user_instruction
        self.idx = idx

    def __lt__(self, other: "HeapDoc") -> bool:
        prompt = get_match_prompt_binary(self.doc, other.doc, self.user_instruction, strategy=self.strategy)
        HeapDoc.num_calls += 1
        HeapDoc.total_tokens += lotus.settings.lm.count_tokens(prompt)
        result: LMOutput = lotus.settings.lm([prompt])
        return parse_ans_binary(result.outputs[0])


def llm_heapsort(
    docs: list[str],
    user_instruction: str,
    K: int,
    strategy: str | None = None,
) -> SemanticTopKOutput:
    """
    Sorts the documents using a heap.

    Args:
        docs (list[str]): The list of documents to sort.
        user_instruction (str): The user instruction for sorting.
        K (int): The number of documents to return.

    Returns:
        SemanticTopKOutput: The indexes of the top k documents and stats.
    """
    HeapDoc.num_calls = 0
    HeapDoc.total_tokens = 0
    HeapDoc.strategy = strategy
    N = len(docs)
    heap = [HeapDoc(docs[idx], user_instruction, idx) for idx in range(N)]
    heap = heapq.nsmallest(K, heap)
    indexes = [heapq.heappop(heap).idx for _ in range(len(heap))]

    stats = {"total_tokens": HeapDoc.total_tokens, "total_llm_calls": HeapDoc.num_calls}
    return SemanticTopKOutput(indexes=indexes, stats=stats)


@pd.api.extensions.register_dataframe_accessor("sem_topk")
class SemTopKDataframe:
    """DataFrame accessor for semantic top k."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        pass

    def __call__(
        self,
        user_instruction: str,
        K: int,
        method: str = "quick",
        strategy: str | None = None,
        group_by: list[str] | None = None,
        cascade_threshold: float | None = None,
        return_stats: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """
        Sorts the DataFrame based on the user instruction and returns the top K rows.

        Args:
            user_instruction (str): The user instruction for sorting.
            K (int): The number of rows to return.
            method (str): The method to use for sorting. Options are "quick", "heap", "naive", "quick-sem".
            group_by (list[str] | None): The columns to group by before sorting. Each group will be sorted separately.
            cascade_threshold (float | None): The confidence threshold for cascading to a larger model.
            return_stats (bool): Whether to return stats.

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The sorted DataFrame. If return_stats is True, returns a tuple with the sorted DataFrame and stats
        """
        lotus.logger.debug(f"Sorting DataFrame with user instruction: {user_instruction}")
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(f"Columns: {col_li}")

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        # Separate code path for grouping
        if group_by:
            grouped = self._obj.groupby(group_by)
            new_df = pd.DataFrame()
            stats = {}
            for name, group in grouped:
                res = group.sem_topk(
                    user_instruction,
                    K,
                    method=method,
                    strategy=strategy,
                    group_by=None,
                    cascade_threshold=cascade_threshold,
                    return_stats=return_stats,
                )

                if return_stats:
                    sorted_group, group_stats = res
                    stats[name] = group_stats
                else:
                    sorted_group = res

                new_df = pd.concat([new_df, sorted_group])

            if return_stats:
                return new_df, stats
            return new_df

        if method == "quick-sem":
            assert len(col_li) == 1, "Only one column can be used for embedding optimization"
            col_name = col_li[0]
            # Sort the dataframe by the column to be used for embedding optimization
            self._obj = self._obj.sem_index(col_name, f"{col_name}_lotus_index").sem_search(
                col_name, user_instruction, len(self._obj)
            )

        df_txt = task_instructions.df2text(self._obj, col_li)
        lotus.logger.debug(f"df_txt: {df_txt}")
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        if method in ["quick", "quick-sem"]:
            output = llm_quicksort(
                df_txt,
                formatted_usr_instr,
                K,
                embedding=method == "quick-sem",
                strategy=strategy,
                cascade_threshold=cascade_threshold,
            )
        elif method == "heap":
            output = llm_heapsort(df_txt, formatted_usr_instr, K, strategy=strategy)
        elif method == "naive":
            output = llm_naive_sort(
                df_txt,
                formatted_usr_instr,
                strategy=strategy,
            )
        else:
            raise ValueError(f"Method {method} not recognized")

        new_df = self._obj.reset_index(drop=True)
        new_df = new_df.reindex(output.indexes).reset_index(drop=True)
        new_df = new_df.head(K)
        if return_stats:
            return new_df, output.stats
        return new_df
