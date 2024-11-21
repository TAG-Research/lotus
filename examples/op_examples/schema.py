from datasets import load_dataset

import lotus
from lotus.models import LM

lm = LM(model="ollama/llama3.1")

lotus.settings.configure(lm=lm)

dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
df = dataset.to_pandas().head(3)

columns = ["problem", "dataset", "results"]

user_instruction = "{abstract}"
new_df = df.sem_to_schema(user_instruction, columns)
print(new_df)
