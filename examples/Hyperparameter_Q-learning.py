import gym
import gym_sokoban
import numpy as np
import random

env_name = 'Boxoban-Test-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))
q_table = np.zeros([5000, env.action_space.n])

num = 44

alpha = 0.3
gamma = 0.9
epsilon = 0.9
epochs = 1000

count = 0
count_success = 0
pre_reward = 0
highest_reward = 0

for i_episode in range(epochs):
    observation = env.reset()
    done = False

    count += 1
    print("No. of episode", count)

    while not done:
        # if count == 1:
            # env.render(mode='human')

        random_value = random.uniform(0, 1)
        if random_value < epsilon:
            action = env.action_space.sample()  # Explore a random action
        else:
            action = np.argmax(q_table[observation])  # Use the action with the highest q-value

        if action not in range(0, 9):
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        # print(ACTION_LOOKUP[action], reward, done, info)

        if not done:
            pre_reward = reward

        # 가장 큰 리워드 계산
        # if reward > highest_reward:
            # highest_reward = reward

        prev_q = q_table[observation, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - alpha) * prev_q + alpha * (reward + gamma * next_max_q)
        q_table[observation, action] = new_q

        observation = next_state

    if pre_reward < reward:
        count_success += 1

np.savetxt("/Users/rosa/Desktop/SELAB/git/sokoban-results/"
           + "discount-factor/"
           + str(num) + "-room03-" + str(alpha) + "-" + str(gamma) + "-" + str(epsilon) + "-" + str(count) +
           "-" + str(count_success) + "-" + str(highest_reward) + ".txt", q_table, fmt="%s")

print("Training finished.")
print("Total episodes : ", count)

print(f"success count: {count_success}")
print(f"success highest reward: {highest_reward}")
print("Total 학습율 : ", count_success/count)


