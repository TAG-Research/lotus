# LOTUS:  An Engine For Semantic Queries
<!--- BADGES: START --->
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OzoJXH13aOwNOIEemClxzNCNYnqSGxVl?usp=sharing)
[![Arxiv](https://img.shields.io/badge/arXiv-2407.11418-B31B1B.svg)][#arxiv-paper-package]
[![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)](https://lotus-ai.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)][#pypi-package]
[![PyPI](https://img.shields.io/pypi/v/lotus-ai)][#pypi-package]
[![GitHub license](https://img.shields.io/badge/License-MIT-blu.svg)][#license-gh-package]

[#license-gh-package]: https://lbesson.mit-license.org/
[#arxiv-paper-package]: TODO
[#docs-package]:TODO
[#pypi-package]: TODO
[#conda-forge-package]: TODO
<!--- BADGES: END --->

Easily build knowledge-intensive LLM applications that reason over your data with LOTUS!

LOTUS provides a declarative programming model and an optimized query engine for serving powerful reasoning-based query pipelines over structured and unstructured data! We provide a simple and intuitive Pandas-like API, that implements **semantic operators** to extend the relational model with a set of modular language-based operators. Programmers can easily compose semantic operators along with traditional data operations to build state-of-the-art AI systems that reason over vast knowledge corpora.

Below are just a few semantic operators provided by LOTUS. For more details, check out the full documentation [documentation](), or [colab tutorial](https://colab.research.google.com/drive/1OzoJXH13aOwNOIEemClxzNCNYnqSGxVl?usp=sharing)

| Operator   | Description                                     |
|------------|-------------------------------------------------|
| sem_map    | Map each row of the dataframe using a natural language projection                   |
| sem_filter | Keep rows that match the natural language predicate                |
| sem_agg    | Performs a natural language aggregation across all rows of a column (e.g. for summarization)           |
| sem_topk   | Order the dataframe by some natural langauge sorting criteria            |
| sem_join   | Join two dataframes based on a natural language predicate        |
| sem_index  | Create a semantic similarity index over a text column           |
| sem_search | Perform top-k search the over a text column          |


# Installation
```
conda create -n lotus python=3.9 -y
conda activate lotus
pip install lotus-ai
```

# Quickstart
If you're already familiar with Pandas, getting started will be a breeze! Below we provide a simple example program using the semantic join operator. The join, like many semantic operators, are specified by **langex** (natural language expressions), which the programmer uses to specify the operation. Each langex is paramterized by one or more table columns, denoted in brackets. The join's langex serves as a predicate and is parameterized by a right and left join key.
```python
import pandas as pd
import lotus
from lotus.models import OpenAIModel

# configure the LM, and remember to export your API key
lm = OpenAIModel()
lotus.settings.configure(lm=lm)

# create dataframes with course names and skills
courses_data = {
    "Course Name": [
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}
skills_data = {"Skill": ["Math", "Computer Science"]}
courses_df = pd.DataFrame(courses_data)
skills_df = pd.DataFrame(skills_data)

# lotus sem join 
res = courses_df.sem_join(skills_df, "Taking {Course Name:left} will help me learn {Skill:right}")
print(res)
```

# Citation
If you use LOTUS in a research paper, please cite this work as follows:
```bibtex
@misc{patel2024lotusenablingsemanticqueries,
      title={LOTUS: Enabling Semantic Queries with LLMs Over Tables of Unstructured and Structured Data},
      author={Liana Patel and Siddharth Jha and Carlos Guestrin and Matei Zaharia},
      year={2024},
      eprint={2407.11418},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2407.11418},
}
```
