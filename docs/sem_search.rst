Semantic Search
==================

.. automodule:: lotus.sem_ops.sem_search
    :members:
    :show-inheritance:

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
| 13| Introduction to Computer Networks       |
| 4 | Introduction to Computer Science        |
| 5 | Introduction to Data Science            |
+---+-----------------------------------------+