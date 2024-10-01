import pandas as pd
import pytest

import lotus
from lotus.models import CrossEncoderModel, E5Model

# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")


@pytest.fixture
def setup_models():
    # Set up embedder and reranker model
    rm = E5Model(model="intfloat/e5-small-v2")
    reranker = CrossEncoderModel(model="mixedbread-ai/mxbai-rerank-xsmall-v1")
    return rm, reranker


def test_cluster_by(setup_models):
    rm, _ = setup_models
    lotus.settings.configure(rm=rm)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "index_dir")
    df = df.sem_cluster_by("Course Name", 2)
    groups = df.groupby("cluster_id")["Course Name"].apply(set).to_dict()
    assert len(groups) == 2, groups
    if "Cooking" in groups[0]:
        cooking_group = groups[0]
        probability_group = groups[1]
    else:
        cooking_group = groups[1]
        probability_group = groups[0]

    assert cooking_group == {"Cooking", "Food Sciences"}, groups
    assert probability_group == {"Probability and Random Processes", "Optimization Methods in Engineering"}, groups


def test_search_rm_only(setup_models):
    rm, _ = setup_models
    lotus.settings.configure(rm=rm)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "index_dir")
    df = df.sem_search("Course Name", "Optimization", K=1)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering"]


def test_search_reranker_only(setup_models):
    _, reranker = setup_models
    lotus.settings.configure(reranker=reranker)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_search("Course Name", "Optimization", n_rerank=2)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering", "Probability and Random Processes"]


def test_search(setup_models):
    rm, reranker = setup_models
    lotus.settings.configure(rm=rm, reranker=reranker)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "index_dir")
    df = df.sem_search("Course Name", "Optimization", K=2, n_rerank=1)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering"]


def test_dedup(setup_models):
    rm, _ = setup_models
    lotus.settings.configure(rm=rm)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Probability and Markov Chains",
            "Harry Potter",
            "Harry James Potter",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.85)
    kept = df["Text"].tolist()
    kept.sort()
    assert len(kept) == 2, kept
    assert "Harry" in kept[0], kept
    assert "Probability" in kept[1], kept


def test_sim_join(setup_models):
    rm, _ = setup_models
    lotus.settings.configure(rm=rm)

    data1 = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
        ]
    }

    data2 = {"Skill": ["Math", "History"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2).sem_index("Skill", "index_dir")
    joined_df = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
    joined_pairs = set(zip(joined_df["Course Name"], joined_df["Skill"]))
    expected_pairs = {("History of the Atlantic World", "History"), ("Riemannian Geometry", "Math")}
    assert joined_pairs == expected_pairs, joined_pairs
