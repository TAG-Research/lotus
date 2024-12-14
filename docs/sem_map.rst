sem_map
=================

.. automodule:: lotus.sem_ops.sem_map
    :members:
    :show-inheritance:

Overview
----------
Similar to the Extract operator, sem_map projects to an arbitrary text attribute rather 
than each tuple to a list of sub-strings.

Motivation
-----------
Traditional data mapping methods often rely on rigid, predefined rules or exact string matches, 
which limits their ability to handle nuanced or context-sensitive transformations. The sem_map operator 
leverages language models to perform context-aware projections, making it a powerful tool for generating 
insights, augmenting data, and enabling intelligent transformations.

Example
----------
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
    user_instruction = "What is a similar course to {Course Name}. Be concise."
    df = df.sem_map(user_instruction)
    print(df)

Output:

+---+----------------------------------------+----------------------------------------------------------------+
|   | Course Name                            | _map                                                           |
+===+========================================+================================================================+
| 0 | Probability and Random Processes       | A similar course to "Probability and Random Processes"...      |
+---+----------------------------------------+----------------------------------------------------------------+
| 1 | Optimization Methods in Engineering    | A similar course to "Optimization Methods in Engineering"...   |
+---+----------------------------------------+----------------------------------------------------------------+
| 2 | Digital Design and Integrated Circuits | A similar course to "Digital Design and Integrated Circuits"...|
+---+----------------------------------------+----------------------------------------------------------------+
| 3 | Computer Security                      | A similar course to "Computer Security" is "Cybersecurity"...  |
+---+----------------------------------------+----------------------------------------------------------------+

Required Parameters
---------------------
- **user_instruction** : The user instruction for map.
- **postprocessor** : The postprocessor for the model outputs. Defaults to map_postprocess.

Optional Parameters
---------------------
- **return_explanations** : Whether to return explanations. Defaults to False.
- **return_raw_outputs** : Whether to return raw outputs. Defaults to False.
- **suffix** : The suffix for the new columns. Defaults to "_map".
- **examples** : The examples dataframe. Defaults to None.
- **strategy** : The reasoning strategy. Defaults to None.