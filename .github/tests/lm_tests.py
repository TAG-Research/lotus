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
    data = {"Text": ["I am really excited to go to class today!", "I am very sad"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"
    filtered_df = df.sem_filter(user_instruction)

    expected_df = pd.DataFrame({"Text": ["I am really excited to go to class today!"]})
    assert filtered_df.equals(expected_df)


def test_filter_cascade(setup_models):
    gpt_4o_mini, gpt_4o = setup_models
    lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

    data = {
        "Text": [
            # Positive examples
            "I am really excited to go to class today!",
            "Today is going to be an amazing day!",
            "I absolutely love the new project I am working on.",
            "Feeling so grateful for everything I have.",
            "I can't wait to see my friends this weekend!",
            "The weather is beautiful, and I feel fantastic.",
            "Just received some great news about my promotion!",
            "I'm so happy to have such supportive colleagues.",
            "I'm thrilled to be learning something new every day.",
            "Life is really good right now, and I feel blessed.",
            "I am proud of all the progress I've made this year.",
            "Today was productive, and I feel accomplished.",
            "I’m really enjoying my workout routine lately!",
            "Got a compliment from my manager today, feeling awesome!",
            "Looking forward to spending time with family tonight.",
            "Just finished a great book and feel inspired!",
            "Had a lovely meal with friends, life is good!",
            "Everything is going as planned, couldn't be happier.",
            "Feeling super motivated and ready to take on challenges!",
            "I appreciate all the small things that bring me joy.",

            # Negative examples
            "I am very sad.",
            "Today has been really tough; I feel exhausted.",
            "I'm feeling pretty down about how things are going.",
            "I’m overwhelmed with all these challenges.",
            "It’s hard to stay positive when things keep going wrong.",
            "I feel so alone and unappreciated.",
            "My energy is low, and nothing seems to cheer me up.",
            "Feeling anxious about everything lately.",
            "I’m disappointed with the way my project turned out.",
            "Today has been one of those days where everything goes wrong.",
            "Life feels really overwhelming right now.",
            "I can't seem to find any motivation these days.",
            "I’m worried about the future and what it holds.",
            "It's been a stressful day, and I feel mentally drained.",
            "I feel like I'm falling behind everyone else.",
            "Just can't seem to catch a break recently.",
            "I’m really struggling to keep up with all my responsibilities.",
            "Had an argument with a close friend, feeling hurt.",
            "I don’t feel supported by my team at work.",
            "Life has been tough lately, and I’m feeling down.",
        ]
    }

    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"

    # All filters resolved by the helper model
    filtered_df, stats = df.sem_filter(
        user_instruction=user_instruction,
        learn_cascade_threshold_sample_percentage=0.5,
        recall_target=0.9,
        precision_target=0.9,
        failure_probability=0.2,
        return_stats=True,
    )

    assert "I am really excited to go to class today!" in filtered_df["Text"].values
    assert "I am very sad" not in filtered_df["Text"].values
    assert stats["filters_resolved_by_helper_model"] > 0, stats


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
