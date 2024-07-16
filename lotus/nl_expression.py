import re
from typing import List

# create a type natural_language_expr which inherits from type str
natural_language_expr = type("natural_language_expr", (str,), {})


def parse_cols(text):
    # Regular expression pattern to match variables in brackets not escaped by double brackets
    pattern = r"(?<!\{)\{(?!\{)(.*?)(?<!\})\}(?!\})"
    # Find all matches in the text
    matches = re.findall(pattern, text)
    return matches


def nle2str(nle: natural_language_expr, cols: List[str]) -> str:
    dict = {}
    for col in cols:
        dict[col] = f"{col.capitalize()}"
    return nle.format(**dict)


# Example usage:
text = "This is a {test} string with {variable} and {{escaped_variable}}."
assert parse_cols(text) == [
    "test",
    "variable",
], f"parse_cols(text) = {parse_cols(text)}"
