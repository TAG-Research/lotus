Semantic Similarity Join
=========================

.. automodule:: lotus.sem_ops.sem_sim_join
    :members:
    :show-inheritance:

Example
---------
.. code-block:: python
    import pandas as pd

    import lotus
    from lotus.models import LM, LiteLLMRM

    lm = LM(model="gpt-4o-mini")
    rm = LiteLLMRM(model="text-embedding-3-small")

    lotus.settings.configure(lm=lm, rm=rm)
    data = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
            "Operating Systems",
            "Food Science",
            "Compilers",
            "Intro to computer science",
        ]
    }

    data2 = {"Skill": ["Math", "Computer Science"]}

    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2).sem_index("Skill", "skill_index")
    res = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
    print(res)

Output
+---+------------------------------+----------+-------------------+
|   |         Course Name          | _scores  |       Skill       |
+---+------------------------------+----------+-------------------+
| 0 | History of the Atlantic World| 0.107831 | Math              |
| 1 | Riemannian Geometry          | 0.345694 | Math              |
| 2 | Operating Systems            | 0.426621 | Computer Science  |
| 3 | Food Science                 | 0.431801 | Computer Science  |
| 4 | Compilers                    | 0.345494 | Computer Science  |
| 5 | Intro to computer science    | 0.676943 | Computer Science  |
+---+------------------------------+----------+-------------------+