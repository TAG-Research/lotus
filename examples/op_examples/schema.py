from datasets import load_dataset

import lotus
from lotus.models import LM

lm = LM(model="ollama/llama3.1")

lotus.settings.configure(lm=lm)

dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
df = dataset.to_pandas().head(3)

columns = ["problem", "dataset", "results"]
col_descriptions = ["Description of the problem", "What dataset is used", "What results are obtained"]

user_instruction = "{abstract}"
new_df = df.sem_schema(user_instruction, columns, col_descriptions)
print(new_df)
