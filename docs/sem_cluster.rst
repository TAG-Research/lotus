sem_cluster_by
=====================

Overview
---------
The cluster operator creates groups over the input dataframe according
to the langex grouping criteria by clustering the data into a target
number of groups. By default, the operator automatically
selects group labels, but the user can also optionally specify target
labels. This operator is useful both for unsupervised discovery of
semantic groups, and for semantic classification tasks.

Motivation
-----------
Performing this semantic operator entails discovering group labels
and classifying each input tuple using the discovered labels. In general, 
performing the unsupervised group discovery is a clustering
task, which is NP-hard. Our algorithm instead offers a
tractable implementation to obtain group labels using a linear pass
over the data with the reference model

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

Output:

+---+----------------------------------------+------------+
|   |           Course Name                  | cluster_id |
+---+----------------------------------------+------------+
| 0 | Probability and Random Processes       | 0          |
+---+----------------------------------------+------------+
| 1 | Optimization Methods in Engineering    | 0          |
+---+----------------------------------------+------------+
| 2 | Digital Design and Integrated Circuits | 0          |
+---+----------------------------------------+------------+
| 3 | Computer Security                      | 1          |
+---+----------------------------------------+------------+
| 4 | Cooking                                | 1          |
+---+----------------------------------------+------------+
| 5 | Food Sciences                          | 1          |
+---+----------------------------------------+------------+


Required Parameters
--------------------
- **col_name** : The column name to cluster on.
- **ncentroids** : The number of centroids.

Optional Parameters
---------------------
- **niter** : The number of iterations.
- **verbose** : Whether to print verbose output.