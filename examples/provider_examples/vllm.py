import pandas as pd

import lotus
from lotus.models import OpenAIModel

lm = OpenAIModel(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    api_base="http://localhost:8000/v1",
    provider="vllm",
)

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
user_instruction = "{Course Name} requires a lot of math"
df = df.sem_filter(user_instruction)
print(df)
