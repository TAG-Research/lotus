Semantic Map
=================

.. automodule:: lotus.sem_ops.sem_map
    :members:
    :show-inheritance:

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

Output
+---+----------------------------------------+----------------------------------------------------------------+
|   |           Course Name                  |                          _map                                  |
+---+----------------------------------------+----------------------------------------------------------------+
| 0 | Probability and Random Processes       | A similar course to "Probability and Random Processes"...      |
| 1 | Optimization Methods in Engineering    | A similar course to "Optimization Methods in Engineering"...   |
| 2 | Digital Design and Integrated Circuits | A similar course to "Digital Design and Integrated Circuits"...|
| 3 | Computer Security                      | A similar course to "Computer Security" is "Cybersecurity"...  |
+---+----------------------------------------+----------------------------------------------------------------+