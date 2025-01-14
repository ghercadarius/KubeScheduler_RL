import json
import os
import asyncio
import torch
from env import KubernetesEnv
from dqn import DQNAgent
import uuid
import aiofiles
from concurrent.futures import ProcessPoolExecutor

import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("asyncio").setLevel(logging.DEBUG)


# Function to load configuration from a file
def load_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


# Function to train and save the model
async def train_model(config):
    env = KubernetesEnv(num_nodes=config["num_nodes"], response_timeout=config["response_timeout"])
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    episodes = config["episodes"]
    batch_size = config["batch_size"]

    for e in range(episodes):
        state = env.reset()
        for time in range(50):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e + 1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    torch.save(agent.model.state_dict(), f"./models/{config["model_file"]}")
    print(f"Model saved to {config['model_file']}")


async def run_model(config, config_index, results_folder):
    env = KubernetesEnv(num_nodes=config["num_nodes"], response_timeout=config["response_timeout"])
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(config["model_file"]))
    agent.model.eval()

    state = env.reset()

    async with aiofiles.open(f"{results_folder}/res_{config_index}", "w") as f:
        for time in range(100):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            print(f"Step {time}: Action {action}, Reward {reward}")
            state = next_state

            if done:
                for pod in env.pods:
                    print(pod.getAssignedNode(), sep=" ")
                    await f.write(f"{pod.getAssignedNode()}\n")
                print("Nodes:")
                await f.write("Nodes: \n")
                for node in env.nodes:
                    print(node)
                    await f.write(f"{node}\n")
                break


# Wrap train_model in a sync function for process execution
def train_model_sync(config):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(train_model(config))
    loop.close()


# Offload training to a separate process
async def train_model_safe(config):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        await loop.run_in_executor(executor, train_model_sync, config)


async def agent_task(config, config_index, results_folder):
    if not os.listdir("./models").__contains__(config["model_file"], ):
        await train_model_safe(config)

    await run_model(config, config_index, results_folder)


async def main():
    config_file = "config_multi.json"
    results_folder = f'results_{uuid.uuid4()}'
    os.makedirs(results_folder, exist_ok=True)
    config = load_config(config_file)
    tasks = []
    if isinstance(config, list):
        for i, c in enumerate(config):
            print(f"Running with configuration {i + 1}")
            print(json.dumps(c, indent=4))
            tasks.append(agent_task(c, i, results_folder))
    else:
        print(f"Running with only one configuration")
        print(json.dumps(config, indent=4))
        tasks.append(agent_task(config, 1, results_folder))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
