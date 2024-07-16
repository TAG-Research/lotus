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

for method in ["quick", "heap", "naive"]:
    sorted_df, stats = df.sem_topk(
        "Which {Course Name} requires the least math?",
        K=2,
        method=method,
        return_stats=True,
    )
    print(sorted_df)
    print(stats)
