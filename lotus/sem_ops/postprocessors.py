import json
from typing import List, Tuple

import lotus


def map_postprocess_cot(llm_answers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Postprocess the output of the map operator with CoT reasoning.

    Args:
        llm_answers (List[str]): The list of llm answers.

    Returns:
        Tuple[List[str], List[str]]: The list of answers and explanations.
    """
    outputs = []
    explanations = []

    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]
        outputs.append(answer)
        explanations.append(reasoning)

    return outputs, explanations


def map_postprocess(llm_answers: List[str], cot_reasoning: bool = False) -> Tuple[List[str], List[str]]:
    """
    Postprocess the output of the map operator.

    Args:
        llm_answers (List[str]): The list of llm answers.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        Tuple[List[str], List[str]]: The list of answers and explanations.
    """
    if cot_reasoning:
        return map_postprocess_cot(llm_answers)

    outputs = llm_answers
    explanations = [None] * len(llm_answers)
    return outputs, explanations


def filter_postprocess_cot(llm_answers: List[str], default: bool) -> Tuple[List[str], List[str]]:
    """
    Postprocess the output of the filter operator with CoT reasoning.

    Args:
        llm_answers (List[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.

    Returns:
        Tuple[List[str], List[str]]: The list of answers and explanations.
    """
    outputs = []
    explanations = []

    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]

        explanations.append(reasoning)

        if "True" in answer:
            outputs.append(True)
        elif "False" in answer:
            outputs.append(False)
        else:
            lotus.logger.info(f"\t Failed to parse: defaulting to {default}")
            outputs.append(default)

    return outputs, explanations


def filter_postprocess(
    llm_answers: List[str],
    default: bool = True,
    cot_reasoning: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Postprocess the output of the filter operator.

    Args:
        llm_answers (List[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        Tuple[List[str], List[str]]: The list of answers and explanations.
    """
    if cot_reasoning:
        return filter_postprocess_cot(llm_answers, default)

    outputs = []
    explanations = [None] * len(llm_answers)
    for answer in llm_answers:
        if "True" in answer:
            outputs.append(True)
        elif "False" in answer:
            outputs.append(False)
        else:
            lotus.logger.info(f"\t Failed to parse: defaulting to {default}")
            outputs.append(default)

    return outputs, explanations


def extract_postprocess(llm_answers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Postprocess the output of the extract operator, which we assume to
    be a JSONL with an answer and quotes field.

    Args:
        llm_answers (List[str]): The list of llm answers.

    Returns:
        Tuple[List[str], List[str]]: The list of answers and quotes.
    """
    answers = []
    quotes = []

    for json_string in llm_answers:
        try:
            data = json.loads(json_string)
            answers.append(data["answer"])
            quotes.append(data["quotes"])
        except Exception as e:
            lotus.logger.error(f"Failed to parse JSON: {e}")
            answers.append(None)
            quotes.append(None)

    return answers, quotes
