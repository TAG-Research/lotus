from typing import List, Optional
import pandas as pd
import re

def extract_image_data(df_text: str) -> (List[str], str):
    """Extracts all instances of base64 image data from df_text and returns them as a list."""
    matches = re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', df_text)
    text_without_images = df_text
    for match in matches:
        text_without_images = text_without_images.replace(match, '')
    return matches, text_without_images.strip()

def filter_formatter_cot(
    df_text: str,
    user_instruction: str,
    examples_df_text: list[str],
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

    for idx in range(len(examples_df_text)):
        ex_df_txt = examples_df_text[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Context:\n{ex_df_txt}\n\nClaim: {user_instruction}",
                },
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nClaim: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })

    return messages


def filter_formatter_zs_cot(
    df_text: str,
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

    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nClaim: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })
        
    return messages


def filter_formatter(
    df_text: str,
    user_instruction: str,
    examples_df_text: list[str] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
) -> list[dict[str, str]]:
    if cot_reasoning:
        assert examples_df_text is not None and examples_answer is not None
        return filter_formatter_cot(df_text, user_instruction, examples_df_text, examples_answer, cot_reasoning)
    elif strategy == "zs-cot":
        return filter_formatter_zs_cot(df_text, user_instruction)

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
                    {
                        "role": "user",
                        "content": f"Context:\n{ex_df_txt}\n\nClaim: {user_instruction}",
                    },
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nClaim: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })
        
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


    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nInstruction: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })

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

    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nInstruction: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })
        
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

    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nInstruction: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })
        
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
    ]

    image_data_list, text_without_images = extract_image_data(df_text)
    usr_image_data_list, usr_instruct_noimg = extract_image_data(user_instruction)
    image_data_list.extend(usr_image_data_list)
    content_entries = [{"type": "image_url", "image_url": {"url": img}} for img in image_data_list]
    content_entries.append({"type": "text", "text": f"Context:\n{text_without_images}\n\nInstruction: {usr_instruct_noimg}"})

    messages.append({
        "role": "user",
        "content": content_entries
    })
        
    return messages


# returns a list of strings corresponding to df rows
def df2text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Formats the given DataFrame into a string containing info from cols."""

    def format_row(x: pd.Series, cols: list[str]) -> str:
        return "".join([f"[{cols[i].capitalize()}]: «{x[cols[i]]}»\n" for i in range(len(cols))])

    # take cols that are in df
    cols = [col for col in cols if col in df.columns]
    formatted_rows: list[str] = df.apply(lambda x: format_row(x, cols), axis=1).tolist()
    return formatted_rows


def li2text(li: list[str], name: str) -> str:
    return "".join([f"[{name}] {li[i]}\n" for i in range(len(li))])
