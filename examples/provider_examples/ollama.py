import pandas as pd

import lotus
from lotus.models import OpenAIModel

lm = OpenAIModel(
    api_base="http://localhost:11434/v1",
    model="llama3.2",
    hf_name="meta-llama/Llama-3.2-3B-Instruct",
    provider="ollama",
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
