import pandas as pd

import lotus
from lotus.models import OpenAIModel

gpt_35_turbo = OpenAIModel("gpt-3.5-turbo")
gpt_4o = OpenAIModel("gpt-4o")

lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_35_turbo)
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
df, stats = df.sem_filter(user_instruction, cascade_threshold=0.95, return_stats=True)
print(df)
print(stats)
