import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import kubeEnv as ke
import defKubeSched as ks
import dql as dql


option = input("Enter 1 for training and saving, 2 for loading and testing\nor 3 for generating new test scenario:")
if option == "1":
    nameFile = input("Enter the name of the file to save the model:")
    nameFile = nameFile + ".pth"
    # Initialize Kubernetes environment
    env = ke.KubernetesEnv(num_nodes=5)
    state_size = len(env.reset())  # State space size
    action_size = env.num_nodes  # Action space size
    agent = dql.DQLAgent(state_size, action_size)
    episodes = 500  # Total training episodes
    batch_size = 32  # Batch size for experience replay
    # Training loop
    for e in range(episodes):
        state = env.reset()  # Reset environment at start of each episode
        for time in range(50):  # Max steps per episode
            action = agent.act(state)  # Agent selects action
            next_state, reward, done, _ = env.step(action)  # Execute action
            # print data about the env
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state  # Move to next state
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break  # End episode if done
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # Train the agent
        for pod in env.deploymentManifests:
            print(pod.getAssignedNode(), sep=" ")
        print()
    torch.save(agent.model.state_dict(), nameFile)
    print("Saved model")
elif option == "2":
    nameFile = input("Enter the name of the file to load the model:")
    nameFile = nameFile
    # Initialize Kubernetes environment
    env = ke.KubernetesEnv(num_nodes=5, readFile=True)
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = dql.DQLAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(nameFile))
    print("Loaded agent")
    agent.model.eval()
    # Testing the trained agent
    state = env.reset()
    # get current date and time in the format: YYYYMMDDHHMMSS
    resultsName = datetime.now().strftime("%Y%m%d%H%M%S") + "DeepQLres.txt"
    testResults = open(resultsName, "w")
    # copy values to test against default scheduler
    print("DQN Scheduler")
    testResults.write("DQN Scheduler\n")
    for time in range(100):  # Test for 20 steps
        # print state
        action = agent.act(state)  # Agent selects action
        next_state, reward, done, _ = env.step(action)  # Execute action
        print(f"Step {time}: Action {action}, Reward {reward}")
        testResults.write(f"Step {time}: Action {action}, Reward {reward}\n")
        state = next_state  # Update state
        if done:
            for pod in env.deploymentManifests:
                print(pod.getAssignedNode(), sep=" ")
                testResults.write(pod.getAssignedNode().__str__() + "\n")
            testResults.write("Nodes:\n")
            print("Nodes:")
            for node in env.nodes:
                testResults.write(node.__str__() + "\n")
                print(node)
            break
    testResults.write("Default Scheduler\n")
    print("Default Scheduler")
    env = ke.KubernetesEnv(num_nodes=5, readFile=True)
    default_scheduler = ks.DefaultScheduler(env)
    for time in range(100):
        classPod = env.deploymentManifests[random.randint(0, len(env.deploymentManifests) - 1)]
        env.pod = classPod.getState()
        action = default_scheduler.schedule(env.pod, classPod)
        testResults.write("Binded pod to node: " + action.__str__() + "\n")
        print("Binded pod to node: ", action)
        if action == -1:
            for pod in env.deploymentManifests:
                testResults.write(pod.getAssignedNode().__str__() + "\n")
                print(pod.getAssignedNode(), sep=" ")
            testResults.write("Nodes:\n")
            print("Nodes:")
            for node in env.nodes:
                testResults.write(node.__str__() + "\n")
                print(node)
            break
    testResults.close()
elif option == "3":
    env = ke.KubernetesEnv(num_nodes=5)
    # GET SAMPLE DATA
    testFile = open("testData.txt", "w")
    testFile.write(len(env.nodes).__str__() + "\n")
    for node in env.nodes:
        testFile.write(node['cpu'].__str__() + "\n")
        testFile.write(node['gpu'].__str__() + "\n")
        testFile.write(node['ram'].__str__() + "\n")
    testFile.write(len(env.deploymentManifests).__str__() + "\n")
    for pod in env.deploymentManifests:
        testFile.write(pod.cpu.__str__() + "\n")
        testFile.write(pod.gpu.__str__() + "\n")
        testFile.write(pod.ram.__str__() + "\n")
        testFile.write(pod.response_time_factor.__str__() + "\n")
    for ind in range(100):
        randIndex = random.randint(0, len(env.deploymentManifests) - 1)
        testFile.write(randIndex.__str__() + " ")
    testFile.write("\n")
    testFile.close()