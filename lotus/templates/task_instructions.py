import pandas as pd
from lotus.utils import fetch_image
from typing import Any

def filter_user_message_formatter(
    multimodal_data: dict[str, Any] | str,
    user_instruction: str,
):
    if isinstance(multimodal_data, str):
        text = multimodal_data
        image_inputs = []

    if isinstance(multimodal_data, dict):
        _image_inputs = [
            [{
                "type": "text",
                "text": f"[{key.capitalize()}]: \n",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url":  base64_image
                },
            }]
            for key, base64_image in multimodal_data["image"].items()
        ]
        image_inputs = [m for image_input in _image_inputs for m in image_input]
        text = multimodal_data["text"] or ""

    return {
        "role": "user", 
        "content": [
            {
            "type": "text",
            "text": f"Claim: {user_instruction}\n\nContext:\n{text}",
            },
        ] + image_inputs
    }
    
    

def filter_formatter_cot(
    multimodal_data: tuple[str, dict[str, str]] | str,
    user_instruction: str,
    examples_multimodal_data: list[str],
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
                filter_user_message_formatter(ex_multimodal_data, user_instruction),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(filter_user_message_formatter(multimodal_data, user_instruction))
    return messages


def filter_formatter_zs_cot(
    multimodal_data: tuple[str, dict[str, str]] | str,
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

    messages.append(filter_user_message_formatter(multimodal_data, user_instruction))
    return messages


def filter_formatter(
    multimodal_data: tuple[str, dict[str, str]] | str,
    user_instruction: str,
    examples_df_text: list[str] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_df_text is not None and examples_answer is not None
        return filter_formatter_cot(multimodal_data, user_instruction, examples_df_text, examples_answer, cot_reasoning)
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

    if examples_df_text:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_df_text, examples_answer):
            messages.extend(
                [
                    filter_user_message_formatter(ex_df_txt, user_instruction),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    messages.append(filter_user_message_formatter(multimodal_data, user_instruction))
    return messages


def map_formatter_cot(
    df_text: str,
    user_instruction: str,
    examples_df_text: list[str],
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

    for idx in range(len(examples_df_text)):
        ex_df_txt = examples_df_text[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Context:\n{ex_df_txt}\nInstruction: {user_instruction}",
                },
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{df_text}\n\nInstruction: {user_instruction}",
        }
    )
    return messages


def map_formatter_zs_cot(
    df_text: str,
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

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{df_text}\nInstruction: {user_instruction}",
        }
    )
    return messages


def map_formatter(
    df_text: str,
    user_instruction: str,
    examples_df_text: list[str] | None = None,
    examples_answer: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_df_text is not None and examples_answer is not None
        return map_formatter_cot(df_text, user_instruction, examples_df_text, examples_answer, cot_reasoning)
    elif strategy == "zs-cot":
        return map_formatter_zs_cot(df_text, user_instruction)

    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_df_text:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_df_text, examples_answer):
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"Context:\n{ex_df_txt}\n\nInstruction: {user_instruction}",
                    },
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{df_text}\n\nInstruction: {user_instruction}",
        }
    )
    return messages


def extract_formatter(df_text: str, user_instruction: str) -> list[dict[str, str]]:
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to extract the information requested in the instruction.\n"
        "Write the response in JSONL format in a single line with the following fields:\n"
        """{"answer": "your answer", "quotes": "quote from context supporting your answer"}"""
    )
    messages = [
        {"role": "system", "content": sys_instruction},
        {
            "role": "user",
            "content": f"Context:\n{df_text}\n\nInstruction: {user_instruction}",
        },
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
        return [""]*len(df)
    formatted_rows: list[str] = df.apply(lambda x: format_row(x, cols), axis=1).tolist()
    return formatted_rows

def df2multimodal_info(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """
        Formats the given DataFrame into a string containing info from cols. 
        Return a list of dictionaries, each containing text and image data.
    """
    image_cols = [col for col in cols if df[col].dtype == "image"]
    text_cols = [col for col in cols if col not in image_cols]
    text_rows = df2text(df, text_cols)
    multimodal_data = [
        {
            "text": text_rows[i],
            "image": {col.capitalize(): fetch_image(df[col].iloc[i], image_type="base64") for col in image_cols} 
        }
        for i in range(len(df))
    ]
    return multimodal_data

def li2text(li: list[str], name: str) -> str:
    return "".join([f"[{name}] {li[i]}\n" for i in range(len(li))])
