import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o")
lotus.settings.configure(lm=lm)

df = pd.DataFrame(
    {
        "description": [
            "Yoshi is 25 years old",
            "Bowser is 45 years old",
            "Luigi is 15 years old",
        ]
    }
)
input_cols = ["description"]
output_cols = ["name", "age"]

new_df = df.sem_extract(input_cols, output_cols, extract_quotes=True)
print(new_df)
