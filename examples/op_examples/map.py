import pandas as pd

import lotus
from lotus.models import LM

lm = LM()

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
user_instruction = "What is a similar course to {Course Name}. Be concise."
df = df.sem_map(user_instruction)
print(df)
