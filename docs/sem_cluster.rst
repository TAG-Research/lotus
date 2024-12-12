Semantic Clustering
=====================

.. automodule:: lotus.sem_ops.sem_cluster_by
    :members:
    :show-inheritance:

Example
---------
.. code-block:: python
    import pandas as pd

    import lotus
    from lotus.models import LM, SentenceTransformersRM

    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(lm=lm, rm=rm)
    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "Cooking",
            "Food Sciences",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "course_name_index").sem_cluster_by("Course Name", 2)
    print(df)

Output
+---+----------------------------------------+------------+
|   |           Course Name                  | cluster_id |
+---+----------------------------------------+------------+
| 0 | Probability and Random Processes       | 0          |
| 1 | Optimization Methods in Engineering    | 0          |
| 2 | Digital Design and Integrated Circuits | 0          |
| 3 | Computer Security                      | 1          |
| 4 | Cooking                                | 1          |
| 5 | Food Sciences                          | 1          |
+---+----------------------------------------+------------+