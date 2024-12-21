import io
import logging
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import lotus
from lotus.models import LM

logging.getLogger("httpx").setLevel(logging.INFO)

lm = LM(model="gpt-4o-2024-11-20")
lotus.settings.configure(lm=lm)

app = FastAPI()

# Global dictionary to store variables between executions
global_vars: dict[str, Any] = {}


class LotusAction(BaseModel):
    tool: str
    args: dict[str, Any]


class LotusObservation(BaseModel):
    observation: str


def handle_python(code: str) -> str:
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exec(code, global_vars)
        except Exception as e:
            print(str(e), file=stderr)

    output = f"Result of running python code\n\n```python\n{code}\n```\n\n{stdout.getvalue() or stderr.getvalue()}"
    return output


def handle_bash(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = f"Result of running bash command\n\n```bash\n{command}\n```\n\n{result.stdout}"
    return output


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/action")
async def execute_action(action: LotusAction):
    print(f"Executing action: {action}", flush=True)
    if action.tool == "run_python_code":
        output = handle_python(action.args["code"])
        return {"status": "success", "output": output}
    elif action.tool == "run_bash_command":
        output = handle_bash(action.args["command"])
        return {"status": "success", "output": output}

    return {"status": "error", "message": "Unknown tool"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
