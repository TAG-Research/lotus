import pandas as pd

import lotus
from lotus.models import OpenAIModel

lm = OpenAIModel()

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

# 2-shot prompting will be performed with these examples
examples = {
    "Course Name": ["Databases", "Machine Learning"],
    "Answer": ["Cloud Computing", "Computer Vision"],
}
examples_df = pd.DataFrame(examples)

user_instruction = "What is a similar course to {Course Name}?"
df = df.sem_map(user_instruction, examples=examples_df)
print(df)
