Semantic DeDuplication
========================

.. automodule:: lotus.sem_ops.sem_dedup
    :members:
    :show-inheritance:

Example
--------
.. code-block:: python
    import pandas as pd

    import lotus
    from lotus.models import SentenceTransformersRM

    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(rm=rm)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "I don't know what day it is",
            "I don't know what time it is",
            "Harry potter and the Sorcerer's Stone",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.815)
    print(df)

Output
+---+------------------------------------------+
|   |                   Text                   |
+---+------------------------------------------+
| 0 | Probability and Random Processes         |
| 5 | I don't know what time it is             |
| 6 | Harry Potter and the Sorcerer's Stone    |
+---+------------------------------------------+