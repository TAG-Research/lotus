PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python_code",
        "description": (
            "Execute Python code in an environment with pandas, numpy and lotus pre-installed. "
            "The code runs in a persistent session where variables are preserved between executions. "
            "The code is NOT running in a jupyter notebook environment, so you must explicitly print() values to see their output. "
            "You must print out variables to see their output, you cannot just call them at the end of the code."
            "You can use the lotus operators to interact with the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. The code has access to previously defined variables and can use pandas, numpy and lotus packages.",
                },
            },
            "required": ["code"],
        },
    },
}

COMPLETE_TASK_TOOL = {
    "type": "function",
    "function": {"name": "complete_task", "description": "Signal that the agent has completed its task"},
}
