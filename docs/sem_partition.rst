Semantic Partition
====================

.. automodule:: lotus.sem_ops.sem_partition_by
    :members:
    :show-inheritance:

Example
----------
.. code-block:: python
    
    import pandas as pd

    import lotus
    from lotus.models import LM, SentenceTransformersRM

    lm = LM(max_tokens=2048)
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
    df = df.sem_index("Course Name", "course_name_index").sem_partition_by(lotus.utils.cluster("Course Name", 2))
    out = df.sem_agg("Summarize all {Course Name}")._output[0]
    print(out)

