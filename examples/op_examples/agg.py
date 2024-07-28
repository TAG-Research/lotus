import pandas as pd

import lotus
from lotus.models import E5Model, OpenAIModel

lm = OpenAIModel()
rm = E5Model()

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
