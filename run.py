import json
import os
import asyncio
import pickle

import torch
from env import KubernetesEnv
from dqn import DQNAgent
from ql import QAgent
import uuid
import aiofiles
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

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


async def train_ql_model(config):
    env = KubernetesEnv(num_nodes=config["num_nodes"], response_timeout=config["response_timeout"])
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = QAgent(state_size, action_size)
    episodes = config["episodes"]

    for e in range(episodes):
        state = env.reset()
        for time in range(50):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e + 1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break

    with open(f"./models/{config["model_file"]}.pkl", 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("Saved agent")


async def run_model(config, config_index, results_folder):
    env = KubernetesEnv(num_nodes=config["num_nodes"], response_timeout=config["response_timeout"])
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(f"./models/{config["model_file"]}"))
    agent.model.eval()
    state = env.reset()
    async with aiofiles.open(f"{results_folder}/dql/res_{config_index}", "w") as f:
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
                await print_data(env, f)
                break
    return env


async def run_ql_model(config, config_index, results_folder):
    env = KubernetesEnv(num_nodes=config["num_nodes"], response_timeout=config["response_timeout"])
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = QAgent(state_size, action_size)
    with open(f"./models/{config['model_file']}.pkl", 'rb') as f:
        agent.q_table = pickle.load(f)
    print("Loaded agent")
    state = env.reset()
    async with aiofiles.open(f"{results_folder}/ql/res_{config_index}", "w") as f:
        for time in range(100):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            print(f"Step {time}: Action {action}, Reward {reward}")
            state = next_state

            if done:
                for pod in env.pods:
                    print(pod.getAssignedNode(), sep=" ")
                    await f.write(f"{pod.getAssignedNode()}\n")
                print("[QL] Nodes:")
                await f.write("Nodes: \n")

                await print_data(env, f)

                break
    return env


async def print_data(env, f):
    max_node_response_time = 0
    min_node_response_time = 100
    for node in env.nodes:
        print(node)
        if node['response_time'] > max_node_response_time:
            max_node_response_time = node['response_time']
        if node['response_time'] < min_node_response_time:
            min_node_response_time = node['response_time']
        await f.write(f"{node}\n")
    diff = max_node_response_time - min_node_response_time
    await f.write(f"Max response time: {max_node_response_time}\n")
    await f.write(f"Min response time: {min_node_response_time}\n")
    await f.write(f"Diff: {diff}\n")
    # calculate median response time
    response_times = [node['response_time'] for node in env.nodes]
    response_times.sort()
    median = response_times[len(response_times) // 2]
    await f.write(f"Median response time: {median}\n")
    # average
    average = sum(response_times) / len(response_times)
    await f.write(f"Average response time: {average}\n")


# Wrap train_model in a sync function for process execution
def train_model_sync(config):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(train_model(config))
    loop.close()


def train_ql_model_sync(config):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(train_ql_model(config))
    loop.close()


# Offload training to a separate process
async def train_model_safe(config):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        await loop.run_in_executor(executor, train_model_sync, config)


async def train_ql_model_safe(config):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        await loop.run_in_executor(executor, train_ql_model_sync, config)


async def agent_task(config, config_index, results_folder):
    if not os.listdir("./models").__contains__(config["model_file"], ):
        await train_model_safe(config)
    if not os.listdir("./models").__contains__(f"{config['model_file']}.pkl"):
        await train_ql_model_safe(config)

    env_dql = await run_model(config, config_index, results_folder)
    env_ql = await run_ql_model(config, config_index, results_folder)
    plot_response_time(env_dql, env_ql)

#Given two envs, create a plot function that compares the response time of each node in each env
import matplotlib.pyplot as plt

def plot_response_time(env_dql, env_ql):
    # Create a dictionary to store response times for each environment
    response_times_dql = {}
    response_times_ql = {}

    # Populate response times for env_dql
    for i, node in enumerate(env_dql.nodes):
        response_times_dql[i] = node['response_time']

    # Populate response times for env_ql
    for i, node in enumerate(env_ql.nodes):
        response_times_ql[i] = node['response_time']

    # Plot response times for env_dql (blue)
    plt.plot(
        list(response_times_dql.keys()),
        list(response_times_dql.values()),
        marker='o',
        color='blue',
        label='DQL Environment'
    )

    # Plot response times for env_ql (red)
    plt.plot(
        list(response_times_ql.keys()),
        list(response_times_ql.values()),
        marker='x',
        color='red',
        label='QL Environment'
    )

    # Customize the plot
    plt.xlabel("Node Index")
    plt.ylabel("Response Time")
    plt.title("Response Times per Node")
    plt.legend()
    plt.xticks(
        ticks=list(response_times_dql.keys()),
        labels=list(response_times_dql.keys())
    )
    plt.grid(True)
    plt.show()

async def main():
    config_file = "config_multi.json"
    results_folder = f'results_{uuid.uuid4()}'
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/dql", exist_ok=True)
    os.makedirs(f"{results_folder}/ql", exist_ok=True)
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
