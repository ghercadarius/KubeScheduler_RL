import json
import os

import torch
from env import KubernetesEnv
from dqn import DQNAgent
from scheduler import DefaultScheduler
import random

# Function to load configuration from a file
def load_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config

# Function to train and save the model
def train_model(config):
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
                print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    torch.save(agent.model.state_dict(), config["model_file"])
    print(f"Model saved to {config['model_file']}")

# Function to test the model
def run_model(config):
    env = KubernetesEnv(num_nodes=config["num_nodes"], response_timeout=config["response_timeout"])
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(config["model_file"]))
    agent.model.eval()

    state = env.reset()
    for time in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        print(f"Step {time}: Action {action}, Reward {reward}")
        state = next_state

        if done:
            for pod in env.pods:
                print(pod.getAssignedNode(), sep=" ")
            print("Nodes:")
            for node in env.nodes:
                print(node)
            break

# Main function to run the training and testing
def main():
    config_file = "config_single.json"  # Path to the configuration file
    config = load_config(config_file)

    print("Running with the following configuration:")
    print(json.dumps(config, indent=4))
    if not os.listdir(".").__contains__(config["model_file"]):
        train_model(config)  # Train the model
    run_model(config)   # Test the model

if __name__ == "__main__":
    main()