import pandas as pd
import pytest

import lotus
from lotus.models import OpenAIModel

# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")


@pytest.fixture
def setup_models():
    # Setup GPT models
    gpt_4o_mini = OpenAIModel(model="gpt-4o-mini")
    gpt_4o = OpenAIModel(model="gpt-4o")
    return gpt_4o_mini, gpt_4o


def test_filter_operation(setup_models):
    gpt_4o_mini, _ = setup_models
    lotus.settings.configure(lm=gpt_4o_mini)

    # Test filter operation on an easy dataframe
    data = {"Text": ["I am really exicted to go to class today!", "I am very sad"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"
    filtered_df = df.sem_filter(user_instruction)

    expected_df = pd.DataFrame({"Text": ["I am really exicted to go to class today!"]})
    assert filtered_df.equals(expected_df)


def test_filter_cascade(setup_models):
    gpt_4o_mini, gpt_4o = setup_models
    lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

    data = {"Text": ["I am really exicted to go to class today!", "I am very sad"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"

    # All filters resolved by the helper model
    filtered_df, stats = df.sem_filter(user_instruction, cascade_threshold=0, return_stats=True)
    assert stats["filters_resolved_by_large_model"] == 0, stats
    assert stats["filters_resolved_by_helper_model"] == 2, stats
    expected_df = pd.DataFrame({"Text": ["I am really exicted to go to class today!"]})
    assert filtered_df.equals(expected_df)

    # All filters resolved by the large model
    filtered_df, stats = df.sem_filter(user_instruction, cascade_threshold=1.01, return_stats=True)
    assert stats["filters_resolved_by_large_model"] == 2, stats
    assert stats["filters_resolved_by_helper_model"] == 0, stats
    assert filtered_df.equals(expected_df)


def test_top_k(setup_models):
    gpt_4o_mini, _ = setup_models
    lotus.settings.configure(lm=gpt_4o_mini)

    data = {
        "Text": [
            "Lionel Messi is a good soccer player",
            "Michael Jordan is a good basketball player",
            "Steph Curry is a good basketball player",
            "Tom Brady is a good football player",
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "Which {Text} is most related to basketball?"
    sorted_df = df.sem_topk(user_instruction, K=2)

    top_2_expected = set(["Michael Jordan is a good basketball player", "Steph Curry is a good basketball player"])
    top_2_actual = set(sorted_df["Text"].values)
    assert top_2_expected == top_2_actual


def test_join(setup_models):
    gpt_4o_mini, _ = setup_models
    lotus.settings.configure(lm=gpt_4o_mini)

    data1 = {"School": ["UC Berkeley", "Stanford"]}
    data2 = {"School Type": ["Public School", "Private School"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    join_instruction = "{School} is a {School Type}"
    joined_df = df1.sem_join(df2, join_instruction)
    joined_pairs = set(zip(joined_df["School"], joined_df["School Type"]))
    expected_pairs = set([("UC Berkeley", "Public School"), ("Stanford", "Private School")])
    assert joined_pairs == expected_pairs


def test_join_cascade(setup_models):
    gpt_4o_mini, gpt_4o = setup_models
    lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

    data1 = {"School": ["UC Berkeley", "Stanford"]}
    data2 = {"School Type": ["Public School", "Private School"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    join_instruction = "{School} is a {School Type}"
    expected_pairs = set([("UC Berkeley", "Public School"), ("Stanford", "Private School")])

    # All joins resolved by the helper model
    joined_df, stats = df1.sem_join(df2, join_instruction, cascade_threshold=0, return_stats=True)
    joined_pairs = set(zip(joined_df["School"], joined_df["School Type"]))
    assert joined_pairs == expected_pairs
    assert stats["filters_resolved_by_large_model"] == 0, stats
    assert stats["filters_resolved_by_helper_model"] == 4, stats

    # All joins resolved by the large model
    joined_df, stats = df1.sem_join(df2, join_instruction, cascade_threshold=1.01, return_stats=True)
    joined_pairs = set(zip(joined_df["School"], joined_df["School Type"]))
    assert joined_pairs == expected_pairs
    assert stats["filters_resolved_by_large_model"] == 4, stats
    assert stats["filters_resolved_by_helper_model"] == 0, stats


def test_map_fewshot(setup_models):
    gpt_4o_mini, _ = setup_models
    lotus.settings.configure(lm=gpt_4o_mini)

    data = {"School": ["UC Berkeley", "Carnegie Mellon"]}
    df = pd.DataFrame(data)
    examples = {"School": ["Stanford", "MIT"], "Answer": ["CA", "MA"]}
    examples_df = pd.DataFrame(examples)
    user_instruction = "What state is {School} in? Respond only with the two-letter abbreviation."
    df = df.sem_map(user_instruction, examples=examples_df, suffix="State")

    pairs = set(zip(df["School"], df["State"]))
    expected_pairs = set([("UC Berkeley", "CA"), ("Carnegie Mellon", "PA")])
    assert pairs == expected_pairs
