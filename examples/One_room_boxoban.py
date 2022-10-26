import gym
import gym_sokoban
import time
import numpy as np
import random
import copy

env_name = 'Boxoban-Test-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

# print("Observation Space: ", env.observation_space)
# print("Action Space       ", env.action_space)

q_table = np.zeros([10000, env.action_space.n])
# a 2D array that represent every possible state and action in the virtual space and initialize all of them to 0
learning_rate = 0.05
discount_factor = 0.7
exploration = 0.7
epochs = 30000

count = 0

for i_episode in range(epochs):
    observation = env.reset()
    done = False

    count += 1
    print("No. of episode", count)

    while not done:
        if (count == 1):
            env.render(mode='human')

        random_value = random.uniform(0, 1)
        if (random_value < exploration):
            action = env.action_space.sample()  # Explore a random action
        else:
            action = np.argmax(q_table[observation])  # Use the action with the highest q-value

        if action not in range(0, 9):
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        # print(ACTION_LOOKUP[action], reward, done, info)

        prev_q = q_table[observation, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q)
        q_table[observation, action] = new_q

        observation = next_state

np.savetxt("25-boxo-0.05-0.7-0.7-30000.txt", q_table, fmt="%s")
print("Training finished.")

total_epochs, total_penalties = 0, 0
episodes = 3  # 3번만 확인

print("Training Evaluation Start.")

count = 0
for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    count += 1
    print("No. of episode", count)
    steps = 0

    while not done:
        env.render(mode='human')
        action = np.argmax(q_table[state])

        # Sleep makes the actions visible for users
        time.sleep(1)
        state, reward, done, info = env.step(action)
        print(ACTION_LOOKUP[action], reward, done, info)
        if reward == -1:
            penalties += 1

        epochs += 1
        steps += 1
    total_penalties += penalties
    total_epochs += epochs
    print(f"steps: {steps}")

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

