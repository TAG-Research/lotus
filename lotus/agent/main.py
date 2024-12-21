import argparse
import os
import subprocess
import time

import requests

from lotus import logger
from lotus.agent.lotus_agent import LotusAction, LotusAgent, LotusObservation


def build_and_launch_container(directory: str):
    # Remove any container with image lotus-agent
    subprocess.run(["docker", "rm", "-f", "lotus-agent"], check=False)

    # Build the docker image
    subprocess.run(
        ["docker", "build", "-t", "lotus-agent", os.path.join(os.path.dirname(__file__), "sandbox")], check=True
    )

    # Create the lotus_agent_output/ directory
    os.makedirs(os.path.join(directory, "lotus_agent_output"), exist_ok=True)

    # Start the container with the mounted directory
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "lotus-agent",
            "-v",
            f"{os.path.abspath(directory)}:/data",
            "-p",
            "8000:8000",
            "-w",
            "/data",
            "-e",
            f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}",
            "lotus-agent",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def send_action(action: LotusAction) -> LotusObservation:
    # Send the action to the container
    response = requests.post("http://localhost:8000/action", json=action.model_dump(), timeout=60)
    return LotusObservation(observation=response.json()["output"])


def wait_for_container():
    # Wait for the container to be ready
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        retry_count += 1
    if retry_count >= max_retries:
        raise TimeoutError("Container failed to start after 30 seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="the directory to run the agent on", required=True)
    parser.add_argument("-t", type=str, help="The task to run", required=True)
    args = parser.parse_args()

    build_and_launch_container(args.d)
    logger.info("Waiting for server to start...")
    wait_for_container()
    logger.info("Server started")

    agent = LotusAgent()
    observations: list[LotusObservation] = []

    action = agent.step(args.t, observations)
    while action.tool != "complete_task":
        logger.info(f"Action: {action}")
        observation = send_action(action)
        logger.info(f"Observation: {observation.observation}")
        observations.append(observation)
        action = agent.step(args.t, observations)


if __name__ == "__main__":
    main()
