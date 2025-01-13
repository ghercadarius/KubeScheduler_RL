from env import KubernetesEnv
from dqn import DQNAgent
from scheduler import DefaultScheduler
import torch
import random

option = input("Enter 1 for training and saving or 2 for loading and testing: ")
if option == "1":
    nameFile = input("Enter the name of the file to save the model: ") + ".pth"
    env = KubernetesEnv(num_nodes=10)
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    episodes = 200
    batch_size = 32
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
    torch.save(agent.model.state_dict(), nameFile)
elif option == "2":
    nameFile = input("Enter the name of the file to load the model:")
    nameFile = nameFile
    # Initialize Kubernetes environment
    env = KubernetesEnv(num_nodes=10)
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(nameFile))
    agent.model.eval()
    # Testing the trained agent
    state = env.reset()
    # copy values to test against default scheduler
    default_state = state.__copy__()
    default_state_size = len(default_state)
    default_action_size = env.num_nodes
    default_nodes = env.nodes
    for time in range(100):  # Test for 20 steps
        # print state
        action = agent.act(state)  # Agent selects action
        next_state, reward, done, _ = env.step(action)  # Execute action
        print(f"Step {time}: Action {action}, Reward {reward}")
        state = next_state  # Update state
        if done:
            for pod in env.pods:
                print(pod.getAssignedNode(), sep=" ")
            print("Nodes:")
            for node in env.nodes:
                print(node)
            break
    default_scheduler = DefaultScheduler(env)
    env.nodes = default_nodes
    for time in range(100):
        action = default_scheduler.schedule(env.pod)
        env.pod = env.pods[random.randint(0, len(env.pods) - 1)].getState()
        print(f"Default Step {time}: Action {action}, Reward {reward}")
        if action == -1 or time == 19:
            for pod in env.pods:
                print(pod.getAssignedNode(), sep=" ")
            print("Nodes:")
            for node in env.nodes:
                print(node)
            break