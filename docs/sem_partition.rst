sem_partition_by
====================

Overview
---------
The sem_partition_by operator in LOTUS enables semantic partitioning of data based on contextual similarities. 
It divides a DataFrame into subsets, which can then be independently analyzed or aggregated.  This operator works 
seamlessly with other LOTUS components, like sem_index for creating embeddings and sem_agg for performing 
aggregations on clustered subsets, to build scalable and efficient workflows.

Motivation
----------
Real-world data often requires grouping based on meaning rather than exact matches, which traditional methods GROUP BY 
cannot handle. The sem_partition_by operator solves this by clustering data semantically, allowing for 
meaningful partitioning of natural language or context-dependent entries.


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


Required Parameters
--------------------
- **partition_fn** : The partitioning function.


