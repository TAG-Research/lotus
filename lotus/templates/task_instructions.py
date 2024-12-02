from typing import Any

import pandas as pd

from lotus.dtype_extensions import ImageDtype


def context_formatter(
    multimodal_data: dict[str, Any] | str,
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(multimodal_data, str):
        text = multimodal_data
        image_inputs: list[dict[str, str]] = []
    elif isinstance(multimodal_data, dict):
        image_data: dict[str, str] = multimodal_data.get("image", {})
        _image_inputs: list[tuple[dict, dict]] = [
            (
                {
                    "type": "text",
                    "text": f"[{key.capitalize()}]: \n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            )
            for key, base64_image in image_data.items()
        ]
        image_inputs = [m for image_input in _image_inputs for m in image_input]
        text = multimodal_data["text"] or ""
    else:
        raise ValueError("multimodal_data must be a dictionary or a string")
    return text, image_inputs


def user_message_formatter(
    multimodal_data: dict[str, Any] | str,
    user_instruction_with_tag: str | None = None,
) -> dict[str, Any]:
    text, image_inputs = context_formatter(multimodal_data)
    if not image_inputs or len(image_inputs) == 0:
        return {
            "role": "user",
            "content": f"Context:\n{text}\n\n{user_instruction_with_tag}",
        }
    content = [{"type": "text", "text": f"Context:\n{text}"}] + image_inputs
    if user_instruction_with_tag:
        content.append({"type": "text", "text": f"\n\n{user_instruction_with_tag}"})
    return {
        "role": "user",
        "content": content,
    }


def filter_formatter_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]],
    examples_answer: list[bool],
    cot_reasoning: list[str],
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'First give your reasoning. Then you MUST end your output with "Answer: True or False"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    for idx in range(len(examples_multimodal_data)):
        ex_multimodal_data = examples_multimodal_data[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                user_message_formatter(ex_multimodal_data, f"Claim: {user_instruction}"),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages


def filter_formatter_zs_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'First give your reasoning. Then you MUST end your output with "Answer: True or False"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages


def filter_formatter(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_multimodal_data is not None and examples_answer is not None
        return filter_formatter_cot(
            multimodal_data, user_instruction, examples_multimodal_data, examples_answer, cot_reasoning
        )
    elif strategy == "zs-cot":
        return filter_formatter_zs_cot(multimodal_data, user_instruction)

    sys_instruction = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'You must answer with a single word, "True" or "False".'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_multimodal_data:
        assert examples_answer is not None
        assert isinstance(examples_multimodal_data, list) and isinstance(examples_answer, list)
        for i in range(len(examples_multimodal_data)):
            ex_multimodal_data = examples_multimodal_data[i]
            ex_ans = examples_answer[i]
            messages.extend(
                [
                    user_message_formatter(ex_multimodal_data, f"Claim: {user_instruction}"),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages


def map_formatter_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]],
    examples_answer: list[str],
    cot_reasoning: list[str],
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        "You must give your reasoning and then your final answer"
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    for idx in range(len(examples_multimodal_data)):
        ex_df_txt = examples_multimodal_data[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def map_formatter_zs_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        'First give your reasoning. Then you MUST end your output with "Answer: your answer"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def map_formatter(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_multimodal_data is not None and examples_answer is not None
        return map_formatter_cot(
            multimodal_data, user_instruction, examples_multimodal_data, examples_answer, cot_reasoning
        )
    elif strategy == "zs-cot":
        return map_formatter_zs_cot(multimodal_data, user_instruction)

    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_multimodal_data:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_multimodal_data, examples_answer):
            messages.extend(
                [
                    user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def extract_formatter(
    multimodal_data: dict[str, Any], output_cols: dict[str, str | None], extract_quotes: bool = True
) -> list[dict[str, str]]:
    output_col_names = list(output_cols.keys())
    # Set the description to be the key if no value is provided
    output_cols_with_desc: dict[str, str] = {col: col if desc is None else desc for col, desc in output_cols.items()}

    all_fields = output_col_names
    if extract_quotes:
        quote_fields = [f"{col}_quote" for col in output_col_names]
        all_fields += quote_fields

    fields_str = ", ".join(all_fields)

    sys_instruction = (
        "The user will provide the columns that need to be extracted and some relevant context.\n"
        f"Your job is to extract these columns and provide only a concise value for each field "
        f"and the corresponding full quote for each field in the '{', '.join([f'{col}_quote' for col in output_col_names])}' fields.\n"
        f"Here is a description of each field: {output_cols_with_desc}\n"
        f"The response should be valid JSON format with the following fields: {fields_str}.\n"
    )

    messages = [
        {"role": "system", "content": sys_instruction},
        user_message_formatter(multimodal_data),
    ]
    return messages


# returns a list of strings corresponding to df rows
def df2text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Formats the given DataFrame into a string containing info from cols."""

    def format_row(x: pd.Series, cols: list[str]) -> str:
        return "".join([f"[{cols[i].capitalize()}]: «{x[cols[i]]}»\n" for i in range(len(cols))])

    # take cols that are in df
    cols = [col for col in cols if col in df.columns]
    if len(cols) == 0:
        return [""] * len(df)
    formatted_rows: list[str] = df.apply(lambda x: format_row(x, cols), axis=1).tolist()
    return formatted_rows


def df2multimodal_info(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """
    Formats the given DataFrame into a string containing info from cols.
    Return a list of dictionaries, each containing text and image data.
    """
    image_cols = [col for col in cols if isinstance(df[col].dtype, ImageDtype)]
    text_cols = [col for col in cols if col not in image_cols]
    text_rows = df2text(df, text_cols)
    multimodal_data = [
        {
            "text": text_rows[i],
            "image": {col.capitalize(): df[col].array.get_image(i, "base64") for col in image_cols},
        }
        for i in range(len(df))
    ]
    return multimodal_data


def merge_multimodal_info(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merges two multimodal info lists into one. Each row of first is merged with each row of second.

    Args:
        first: list of multimodal info dictionaries
        second: list of multimodal info dictionaries

    Returns:
        list of merged multimodal info dictionaries
    """
    return [
        {
            "text": f"{first[i]['text']}\n{second[j]['text']}"
            if first[i]["text"] != "" and second[j]["text"] != ""
            else first[i]["text"] + second[j]["text"],
            "image": {**first[i]["image"], **second[j]["image"]},
        }
        for i in range(len(first))
        for j in range(len(second))
    ]


def li2text(li: list[str], name: str) -> str:
    return "".join([f"[{name}] {li[i]}\n" for i in range(len(li))])
