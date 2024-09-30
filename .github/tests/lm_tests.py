import pandas as pd

import lotus
from lotus.models import OpenAIModel

# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

gpt_4o_mini = OpenAIModel(model="gpt-4o-mini")
gpt_4o = OpenAIModel(model="gpt-4o")
lotus.settings.configure(lm=gpt_4o_mini)

# Test filter operation on an easy dataframe
data = {
    "Text": [
        "I am really exicted to go to class today!",
        "I am very sad",
    ]
}
df = pd.DataFrame(data)
user_instruction = "{Text} is a positive sentiment"
filtered_df = df.sem_filter(user_instruction)

expected_df = pd.DataFrame(
    {
        "Text": [
            "I am really exicted to go to class today!",
        ]
    }
)

assert filtered_df.equals(expected_df), f"Expected {expected_df}\n, but got\n{filtered_df}"

# Test cascade
lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

# All filters are resolved by the large model
filtered_df, stats = df.sem_filter(user_instruction, cascade_threshold=0, return_stats=True)
assert stats["filters_resolved_by_large_model"] == 0
assert stats["filters_resolved_by_helper_model"] == 2
assert filtered_df.equals(expected_df), f"Expected {expected_df}\n, but got\n{filtered_df}"

# All filters are resolved by the helper model
filtered_df, stats = df.sem_filter(user_instruction, cascade_threshold=1, return_stats=True)
assert stats["filters_resolved_by_large_model"] == 2
assert stats["filters_resolved_by_helper_model"] == 0
assert filtered_df.equals(expected_df), f"Expected {expected_df}\n, but got\n{filtered_df}"


# Test top-k on an easy dataframe
lotus.settings.configure(lm=gpt_4o_mini)
data = {
    "Text": [
        "Michael Jordan is a good basketball player",
        "Steph Curry is a good basketball player",
        "Lionel Messi is a good soccer player",
        "Tom Brady is a good football player",
    ]
}
df = pd.DataFrame(data)
user_instruction = "Which {Text} is most related to basketball?"
sorted_df = df.sem_topk(user_instruction, K=2, method="naive")

top_2_expected = set(["Michael Jordan is a good basketball player", "Steph Curry is a good basketball player"])
top_2_actual = set(sorted_df["Text"].values)
assert top_2_expected == top_2_actual, f"Expected {top_2_expected}\n, but got\n{top_2_actual}"

# Test join on an easy dataframe
data1 = {
    "School": [
        "UC Berkeley",
        "Stanford",
    ]
}

data2 = {"School Type": ["Public School", "Private School"]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
join_instruction = "{School} is a {School Type}"
joined_df = df1.sem_join(df2, join_instruction)
joined_pairs = set(zip(joined_df["School"], joined_df["School Type"]))
expected_pairs = set(
    [
        ("UC Berkeley", "Public School"),
        ("Stanford", "Private School"),
    ]
)
assert joined_pairs == expected_pairs, f"Expected {expected_pairs}\n, but got\n{joined_pairs}"

# Test map on an easy dataframe with few-shot examples
data = {
    "School": [
        "UC Berkeley",
        "Carnegie Mellon",
    ]
}
df = pd.DataFrame(data)
examples = {"School": ["Stanford", "MIT"], "Answer": ["CA", "MA"]}
examples_df = pd.DataFrame(examples)
user_instruction = "What state is {School} in? Respond only with the two-letter abbreviation."
df = df.sem_map(user_instruction, examples=examples_df, suffix="State")
pairs = set(zip(df["School"], df["State"]))
expected_pairs = set(
    [
        ("UC Berkeley", "CA"),
        ("Carnegie Mellon", "PA"),
    ]
)
assert pairs == expected_pairs, f"Expected {expected_pairs}\n, but got\n{pairs}"
