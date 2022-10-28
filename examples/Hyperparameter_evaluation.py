import gym
import gym_sokoban as gs
import time
import numpy as np

env_name = 'Boxoban-Test-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

q_table = np.zeros([5000, env.action_space.n])

total_epochs = 0
episodes = 1

print("Training Evaluation Start.")
with open("/Users/rosa/Desktop/SELAB/git/sokoban-results/"
          "epsilon/"
          "96-room03-0.3-0.9-0.96-1000-306-11.4.txt") as textFile:
    q_table1 = [line.split() for line in textFile]
    q_table = np.array(q_table1)

count = 0
count_success = 0
pre_reward = 0
highest_reward = 0
for _ in range(episodes):
    state = env.reset()
    epochs, reward = 0, 0
    done = False

    count += 1
    print("No. of episode", count)
    steps = 0

    while not done:
        if (count == 1):
            env.render(mode='human')
            time.sleep(2)
        action = np.argmax(q_table[state])

        state, reward, done, info = env.step(action)
        print(ACTION_LOOKUP[action], reward, done, info)

        if not done:
            pre_reward = reward

        # 가장 큰 리워드 계산
        if reward > highest_reward:
            highest_reward = reward

        epochs += 1
        steps += 1
    total_epochs += epochs
    if pre_reward < reward:
        count_success += 1

print("")
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
# 학습 후 성공율
print(f"success count: {count_success}")
# print(f"highest reward: {highest_reward}")
print("Total 성공율 : ", count_success/count)



