import json
import os
from typing import Any

from litellm import Message, ModelResponse, completion
from pydantic import BaseModel

from lotus import logger
from lotus.agent.tools import COMPLETE_TASK_TOOL, PYTHON_TOOL


class LotusAction(BaseModel):
    tool: str
    args: dict[str, Any]

    def to_string(self):
        return f"Tool: {self.tool}, Args: {self.args}"


class LotusObservation(BaseModel):
    observation: str

    def to_string(self):
        truncated_observation = self.observation[:5000]
        if len(truncated_observation) < len(self.observation):
            truncated_observation += f"... {len(self.observation) - len(truncated_observation)} more characters"
        return f"Observation: {truncated_observation}"


def parse_response(response: ModelResponse) -> LotusAction:
    try:
        message = response.choices[0].message
        tool_call = message.tool_calls[0]
        tool = tool_call.function.name
        args = tool_call.function.arguments
        return LotusAction(tool=tool, args=json.loads(args))
    except Exception:
        logger.error("Error parsing response, assuming task is complete")
        return LotusAction(tool="complete_task", args={})


class LotusAgent:
    def __init__(self, model_name: str = "gpt-4o-2024-11-20"):
        self.model_name = model_name
        self.system_prompt = self._load_system_prompt()
        self.tools = [PYTHON_TOOL, COMPLETE_TASK_TOOL]
        self.assistant_messages: list[Message] = []

    def _load_system_prompt(self):
        with open(os.path.join(os.path.dirname(__file__), "system_prompt.txt"), "r") as f:
            return f.read()

    def step(self, prompt: str, observations: list[LotusObservation]) -> LotusAction:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        assert len(observations) == len(self.assistant_messages)
        for assistant_message, observation in zip(self.assistant_messages, observations):
            tool_call_id = assistant_message.tool_calls[0].id
            messages.append(json.loads(assistant_message.model_dump_json()))
            messages.append({"role": "tool", "content": observation.to_string(), "tool_call_id": tool_call_id})

        response = completion(model=self.model_name, tools=self.tools, messages=messages)
        self.assistant_messages.append(response.choices[0].message)

        return parse_response(response)
