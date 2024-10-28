# Contributing

## Setting up

To set up for development, create a conda environment, install lotus, and install additional dev dependencies.
```
conda create -n lotus python=3.10 -y
conda activate lotus
git clone git@github.com:stanford-futuredata/lotus.git
pip install -e .
pip install -r requirements-dev.txt
```

## Dev Flow
After making your changes, please make a PR to get your changes merged upstream.

## Running vLLM Models
To use vLLM for model serving, you just need to make an OpenAI compatible vLLM server. Then, the `OpenAIModel` class can be used to point to the server. See an example below.

Create the server
```
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct --port 8000 --tensor-parallel-size 8
```

In LOTUS, you should instantiate your model as follows
```
from lotus.models import OpenAIModel
lm = OpenAIModel(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    api_base="http://localhost:8000/v1",
    provider="vllm",
)
```

## Helpful Examples
For helpful examples of LOTUS operators, please refer to the `examples` folder, as well as the documentation.