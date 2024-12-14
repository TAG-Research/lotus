sem_topk
================

.. automodule:: lotus.sem_ops.sem_topk
    :members:
    :show-inheritance:

Overview
---------
LOTUS supports a semantic top-k, which takes the langex ranking criteria. Programmers can optionally 
specify a group-by parameter to indicate a subset of columns to group over during ranking. 
The groupings are defined using standard equality matches over the group-by columns

Motivation
-----------
Performing semantic top-𝐾 ranking is inherently challenging as it requires logical reasoning across 
rows to determine the most contextually relevant entries. This involves processing large volumes of 
data and capturing subtle relationships that traditional ranking approaches, such as numerical or 
keyword-based methods, often miss. sem_topk overcomes these limitations by leveraging advanced language 
models to evaluate semantic similarity, providing a robust and efficient solution for ranking based on 
natural language queries.

Example
--------
.. code-block:: python
    
    import pandas as pd

    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)
    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
        ]
    }
    df = pd.DataFrame(data)

    for method in ["quick", "heap", "naive"]:
        sorted_df, stats = df.sem_topk(
            "Which {Course Name} requires the least math?",
            K=2,
            method=method,
            return_stats=True,
        )
        print(sorted_df)
        print(stats)


Output:

+---+----------------------------------------+
|   |           Course Name                  |
+---+----------------------------------------+
| 0 | Computer Security                      |
+---+----------------------------------------+
| 1 | Digital Design and Integrated Circuits |
+---+----------------------------------------+

Required Parameters
--------------------
- **user_instruction** : The user instruction for sorting.
- **K**: The number of rows to return.

Optional Paramaters
---------------------
- **method** : The method to use for sorting. Options are "quick", "heap", "naive", "quick-sem".
- **group_by** : The columns to group by before sorting. Each group will be sorted separately.
- **cascade_threshold**: The confidence threshold for cascading to a larger model.
- **return_stats** : Whether to return stats.