import os

import pandas as pd
import pytest
from tokenizers import Tokenizer

import lotus
from lotus.models import LM

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "gpt-4o": ENABLE_OPENAI_TESTS,
    "ollama/llama3.1": ENABLE_OLLAMA_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = LM(model=model_path)

    return models


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()
        model.reset_cache()


################################################################################
# Standard tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_filter_operation(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    data = {"Text": ["I am really excited to go to class today!", "I am very sad"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"
    filtered_df = df.sem_filter(user_instruction)

    expected_df = pd.DataFrame({"Text": ["I am really excited to go to class today!"]})
    assert filtered_df.equals(expected_df)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_top_k(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

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
    top_2_expected = set(["Michael Jordan is a good basketball player", "Steph Curry is a good basketball player"])

    strategies = ["quick", "heap", "naive"]
    for strategy in strategies:
        sorted_df = df.sem_topk(user_instruction, K=2, strategy=strategy)

        top_2_actual = set(sorted_df["Text"].values)
        assert top_2_expected == top_2_actual


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_join(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data1 = {"School": ["UC Berkeley", "Stanford"]}
    data2 = {"School Type": ["Public School", "Private School"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    join_instruction = "{School} is a {School Type}"
    joined_df = df1.sem_join(df2, join_instruction)
    joined_pairs = set(zip(joined_df["School"], joined_df["School Type"]))
    expected_pairs = set([("UC Berkeley", "Public School"), ("Stanford", "Private School")])
    assert joined_pairs == expected_pairs


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_map_fewshot(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {"School": ["UC Berkeley", "Carnegie Mellon"]}
    df = pd.DataFrame(data)
    examples = {"School": ["Stanford", "MIT"], "Answer": ["CA", "MA"]}
    examples_df = pd.DataFrame(examples)
    user_instruction = "What state is {School} in? Respond only with the two-letter abbreviation."
    df = df.sem_map(user_instruction, examples=examples_df, suffix="State")

    # clean up the state names to be more robust to free-form text
    df["State"] = df["State"].str[-2:].str.lower()
    pairs = set(zip(df["School"], df["State"]))
    expected_pairs = set([("UC Berkeley", "ca"), ("Carnegie Mellon", "pa")])
    assert pairs == expected_pairs


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_agg_then_map(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {"Text": ["My name is John", "My name is Jane", "My name is John"]}
    df = pd.DataFrame(data)
    agg_instruction = "What is the most common name in {Text}?"
    agg_df = df.sem_agg(agg_instruction, suffix="draft_output")
    assert len(agg_df) == 1

    map_instruction = "{draft_output} is a draft answer to the question 'What is the most common name?'. Clean up the draft answer so that there is just a single name. Your answer MUST be on word"
    cleaned_df = agg_df.sem_map(map_instruction, suffix="final_output")
    assert cleaned_df["final_output"].values[0].lower().strip(".,!?\"'") == "john"


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_group_by_with_agg(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Names": ["Michael", "Anakin", "Luke", "Dwight"],
        "Show": ["The Office", "Star Wars", "Star Wars", "The Office"],
    }
    df = pd.DataFrame(data)
    agg_instruction = "Summarize {Names}"
    agg_df = df.sem_agg(agg_instruction, suffix="draft_output", group_by=["Show"])
    assert len(agg_df) == 2

    # Map post-processing
    map_instruction = "{draft_output} is a draft answer to the question 'Summarize the names'. Clean up the draft answer is just a comma separated list of names."
    cleaned_df = agg_df.sem_map(map_instruction, suffix="final_output")

    assert set(cleaned_df["final_output"].values[0].lower().strip(".,!?\"'").split(", ")) == {"anakin", "luke"}
    assert set(cleaned_df["final_output"].values[1].lower().strip(".,!?\"'").split(", ")) == {"michael", "dwight"}


################################################################################
# Cascade tests
################################################################################
@pytest.mark.skipif(not ENABLE_OPENAI_TESTS, reason="Skipping test because OpenAI tests are not enabled")
def test_filter_cascade(setup_models):
    models = setup_models
    lotus.settings.configure(lm=models["gpt-4o"], helper_lm=models["gpt-4o-mini"])

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


@pytest.mark.skipif(not ENABLE_OPENAI_TESTS, reason="Skipping test because OpenAI tests are not enabled")
def test_join_cascade(setup_models):
    models = setup_models
    lotus.settings.configure(lm=models["gpt-4o"], helper_lm=models["gpt-4o-mini"])

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


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_format_logprobs_for_filter_cascade(setup_models, model):
    lm = setup_models[model]
    messages = [
        [{"role": "user", "content": "True or False: The sky is blue?"}],
    ]
    response = lm(messages, logprobs=True)
    formatted_logprobs = lm.format_logprobs_for_filter_cascade(response.logprobs)
    true_probs = formatted_logprobs.true_probs
    assert len(true_probs) == 1

    # Very safe (in practice its ~1)
    assert true_probs[0] > 0.8
    assert len(formatted_logprobs.tokens) == len(formatted_logprobs.confidences)


################################################################################
# Token counting tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_count_tokens(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    tokens = lm.count_tokens("Hello, world!")
    assert lm.count_tokens([{"role": "user", "content": "Hello, world!"}]) == tokens
    assert tokens < 100


def test_custom_tokenizer():
    custom_tokenizer = Tokenizer.from_pretrained("gpt2")
    custom_lm = LM(model="doesn't matter", tokenizer=custom_tokenizer)
    tokens = custom_lm.count_tokens("Hello, world!")
    assert custom_lm.count_tokens([{"role": "user", "content": "Hello, world!"}]) == tokens
    assert tokens < 100


################################################################################
# Cache tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_cache(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm, enable_cache=True)

    # Check that "What is the capital of France?" becomes cached
    first_batch = [
        [{"role": "user", "content": "Hello, world!"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]

    first_responses = lm(first_batch).outputs
    assert lm.stats.total_usage.cache_hits == 0

    second_batch = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "What is the capital of Germany?"}],
    ]
    second_responses = lm(second_batch).outputs
    assert second_responses[0] == first_responses[1]
    assert lm.stats.total_usage.cache_hits == 1

    # Test clearing cache
    lm.reset_cache()
    lm.reset_stats()
    lm(second_batch)
    assert lm.stats.total_usage.cache_hits == 0


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_disable_cache(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm, enable_cache=False)

    batch = [
        [{"role": "user", "content": "Hello, world!"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 0
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 0

    # Now enable cache. Note that the first batch is not cached.
    lotus.settings.configure(enable_cache=True)
    first_responses = lm(batch).outputs
    assert lm.stats.total_usage.cache_hits == 0
    second_responses = lm(batch).outputs
    assert lm.stats.total_usage.cache_hits == 2
    assert first_responses == second_responses


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_reset_cache(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm, enable_cache=True)

    batch = [
        [{"role": "user", "content": "Hello, world!"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 0
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 2

    lm.reset_cache(max_size=1)
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 2
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 3

    lm.reset_cache(max_size=0)
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 3
    lm(batch)
    assert lm.stats.total_usage.cache_hits == 3
