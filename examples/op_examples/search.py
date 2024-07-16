import pandas as pd

import lotus
from lotus.models import CrossEncoderModel, E5Model, OpenAIModel

lm = OpenAIModel()
rm = E5Model()
reranker = CrossEncoderModel()

lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Introduction to Computer Science",
        "Introduction to Data Science",
        "Introduction to Machine Learning",
        "Introduction to Artificial Intelligence",
        "Introduction to Robotics",
        "Introduction to Computer Vision",
        "Introduction to Natural Language Processing",
        "Introduction to Reinforcement Learning",
        "Introduction to Deep Learning",
        "Introduction to Computer Networks",
    ]
}
df = pd.DataFrame(data)

df = df.sem_index("Course Name", "index_dir").sem_search(
    "Course Name",
    "Which course name is most related to computer security?",
    K=8,
    n_rerank=4,
)
print(df)
