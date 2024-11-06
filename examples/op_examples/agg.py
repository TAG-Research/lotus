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
df = df.sem_agg("Summarize all {Course Name}")
print(df._output[0])
