import gym
import gym_sokoban
import time
import numpy as np
import random
import copy

env_name = 'Sokoban-small-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

q_table = np.zeros([7000, env.action_space.n])
# a 2D array that represent every possible state and action in the virtual space and initialize all of them to 0
learning_rate = 0.01
discount_factor = 0.7
exploration = 0.5
epochs = 10000

# env 하나 생성해서 복사해놓고 사용
observation = env.reset()
temp_ob = copy.deepcopy(env.reset())
temp = copy.deepcopy(env)
count = 0

np.savetxt("16-save-01-temp_ob.txt", temp_ob, fmt="%s")
print("Saved Observation.")
np.savetxt("16-save-01-temp_env.txt", temp, fmt="%s")
print("Saved temp.")

print("Calling Room Start.")
with open("16-save-01-temp_ob.txt") as textFile:
    temp_ob = [line.split() for line in textFile]
    temp_ob = np.array(temp_ob)

with open("16-save-01-temp_env.txt") as textFile:
    temp = [line.split() for line in textFile]
    temp = np.array(temp)

for i_episode in range(epochs):
    env = copy.deepcopy(temp)
    observation = copy.deepcopy(temp_ob)
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

np.savetxt("16-test.txt", q_table, fmt="%s")
print("Training finished.")

total_epochs, total_penalties = 0, 0
episodes = 3  # 3번만 확인

print("Training Evaluation Start.")
with open("16-test.txt") as textFile:
    q_table1 = [line.split() for line in textFile]
    q_table = np.array(q_table1)

count = 0
for _ in range(episodes):
    env = copy.deepcopy(temp)
    state = copy.deepcopy(temp_ob)
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

