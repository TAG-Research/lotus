Core Concepts
==================

LOTUS' implements the semantic operator programming model. Semantic operators as declarative transformations on one or more
datasets, parameterized by a natural langauge expression (*langex*) that can be implemnted by a variety of AI-based algorithms.
Semantic operators seamlessly extend the relational model, operating over tables that may contain traditional structured data
as well as unstructured fields, such as free-form text. These composable, modular langaiug-based operaters allow you to write 
AI-based piplines with high-level logic, leaving the rest of the work to the query engine! Each operator can be implmented and 
optimized in multiple ways, opening a rich space for execution plans, similar to relational operators. Here is a quick example:

.. code-block:: python

    langex = "The {abstract} suggests that LLMs efficeintly utilize long context"
    filtered_df = papers_df.sem_filter(langex)


With LOTUS, applications can be built by chaining togethor different operators. Much like relational operators can be used to 
transform tables in SQL, LOTUS operators can be use to semantically transform Pandas DataFrames. Here are some key Operators:

+--------------+-----------------------------------------------------+
| Operator     | Description                                         |
+==============+=====================================================+
| Sem_Map      | Map each row of the DataFrame                       |
+--------------+-----------------------------------------------------+
| Sem_Extract  | Extracts attributes and values from a DataFrame     |
+--------------------------------------------------------------------+
| Sem_Filter   | Keep rows that match a predicate                    |
+--------------+-----------------------------------------------------+
| Sem_Agg      | Aggregate information across all rows               |
+--------------+-----------------------------------------------------+
| Sem_TopK     | Order the DataFrame by some criteria                |
+--------------+-----------------------------------------------------+
| Sem_Join     | Join two DataFrames based on a predicate            |
+--------------+-----------------------------------------------------+
| Sem_DeDup    | Deduplicate records based on semantic similarity    |
+--------------+-----------------------------------------------------+
| Sem_Index    | Create a semantic index over a column               |
+--------------+-----------------------------------------------------+
| Sem_Search   | Search the DataFrame for relevant rows              |
+--------------+-----------------------------------------------------+
| Sem_Sim_Join | Join two DataFrames based on Similarity             |
+--------------+-----------------------------------------------------+
| Sem_Cluster  | Clustering on the DataFrame.                        |
+--------------+-----------------------------------------------------+

