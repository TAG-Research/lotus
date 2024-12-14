sem_search
==================

.. automodule:: lotus.sem_ops.sem_search
    :members:
    :show-inheritance:

Overview
----------
Semantic Search assumes a semantic similarity-based ranking criteria relative to a
natural language query. LOTUS also exposes advanced relevancebased re-ranking functionality for search, 
allowing users to specify the n_rerank parameter during the semantic search. The semantic
search in this case will first find the top-𝐾 most relevant documents and then re-rank 
the top-𝐾 found documents to return the top n_rerank.

Motivation
------------
Traditional search relies on exact or partial matches, it often fails to capture nuanced relationships 
and context. The sem_search operator in LOTUS bridges this gap by utilizing advanced language models to 
perform context-aware retrieval and ranking.

Example
-----------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM, CrossEncoderReranker, SentenceTransformersRM

    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")

    lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)
    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "Introduction to Computer Science",
            "Introduction to Data Science",
            "Introduction to Machine Learning",
            "Introduction to Artificial Intelligence",
            "Introduction to Robotics",
            "Introduction to Computer Vision",
            "Introduction to Natural Language Processing",
            "Introduction to Reinforcement Learning",
            "Introduction to Deep Learning",
            "Introduction to Computer Networks",
        ]
    }
    df = pd.DataFrame(data)

    df = df.sem_index("Course Name", "index_dir").sem_search(
        "Course Name",
        "Which course name is most related to computer security?",
        K=8,
        n_rerank=4,
    )
    print(df)

Output

+---+-----------------------------------------+
|   |               Course Name               |
+---+-----------------------------------------+
| 3 | Computer Security                       |
+---+-----------------------------------------+
| 13| Introduction to Computer Networks       |
+---+-----------------------------------------+
| 4 | Introduction to Computer Science        |
+---+-----------------------------------------+
| 5 | Introduction to Data Science            |
+---+-----------------------------------------+

Required Parameters
---------------------
- **col_name** : The column name to search on.
- **query** : The query string.

Optional Parameters
---------------------
- **K**: The number of documents to retrieve.
- **n_rerank** : The number of documents to rerank.
- **return_scores** : Whether to return the similarity scores.
- **suffix** : The suffix to append to the new column containing the similarity scores.
