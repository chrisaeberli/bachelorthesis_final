import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import log
import itertools

#random.seed(1456789)


def calc_reward(arm_rewards, rand):
    reward = 0
    for (pot, prob) in arm_rewards:
        if (rand <= prob):
            reward += pot
    return reward

def simulate(rounds, probs):
    num_arms = len(probs)
    choose_arm = random.randint(0,num_arms-1)
    counter = 0
    last_rewards = [0] * num_arms
    round_counter = 0
    epoch_counter = -1
    total_reward = 0
    for i in rounds:
        epoch_counter += 1
        iterations = rounds[epoch_counter]
        probs_this_epoch = []
        for j in range(num_arms):
            probs_this_epoch.append(probs[j][epoch_counter])
        for j in range(iterations):
            round_counter += 1
            reward = calc_reward(probs_this_epoch[choose_arm], random.random())
            if (reward <= 0.5):
                counter += 1
            total_reward += reward
            if (counter == 3):
                counter = 0
                choose_arm = random.randint(0,num_arms-1)
    return total_reward



count_1 = []
count_2 = []
count_3 = []
count_4 = []
count_5 = []
count_6 = []
count_6 = []
count_7 = []
count_8 = []
count_9 = []
count_10 = []
count_11 = []
count_12 = []
count_13 = []

for i in range(10000):
    print(i)
    count_1.append(simulate([30,30,30], [[[(1,0.9)],[(1,0.1)],[(1,0.9)]],[[(1,0.1)],[(1,0.9)],[(1,0.1)]]]))
    count_2.append(simulate([10,10,10], [[[(1,0.9)],[(1,0.1)],[(1,0.9)]],[[(1,0.1)],[(1,0.9)],[(1,0.1)]]]))
    count_3.append(simulate([30,30,30], [[[(1,0.7)],[(1,0.3)],[(1,0.7)]],[[(1,0.3)],[(1,0.7)],[(1,0.3)]]]))
    count_4.append(simulate([30,30,30], [[[(1,0.6)],[(1,0.4)],[(1,0.6)]],[[(1,0.4)],[(1,0.6)],[(1,0.4)]]]))
    count_5.append(simulate([20,20,20], [[[(1,0.9)],[(1,0.1)],[(1,0.9)]],[[(1,0.1)],[(1,0.9)],[(1,0.1)]],[[(1,0.5)],[(1,0.5)],[(1,0.5)]]]))
    count_6.append(simulate([20,20], [[[(1,0.8)],[(1,0.8)]],[[(1,0)],[(1,1)]]]))
    count_7.append(simulate([30], [[[(0.33,0.75),(0.33,0.5),(0.33,0.25)]],[[(1,0.2)]]]))
    count_8.append(simulate([30], [[[(0.33,0.75),(0.33,0.5),(0.33,0.25)]],[[(1,0.8)]]]))
    count_9.append(simulate([60], [[[(0.33,0.75),(0.33,0.5),(0.33,0.25)]],[[(1,0.2)]]]))
    count_10.append(simulate([60], [[[(0.33,0.75),(0.33,0.5),(0.33,0.25)]],[[(1,0.8)]]]))
    count_11.append(simulate([50,50], [[[(1,0)],[(1,1)]],[[(1,0)],[(1,0)]],[[(1,1)],[(1,0)]],[[(1,0)],[(1,0)]]]))
    count_12.append(simulate([50,50], [[[(1,0)],[(1,1)]],[[(1,0)],[(1,0)]],[[(1,1)],[(1,0)]],[[(1,0)],[(1,0)]],[[(1,0.5)],[(1,0.5)]],[[(1,0.5)],[(1,0.5)]]]))
    count_13.append(simulate([30,30], [[[(0.5,1),(0.5,0)],[(0.5,1),(0.5,0.5)]],[[(0.5,1),(0.5,0.8)],[(0.5,1),(0.5,0.6)]]]))
    
    
print(np.average(count_1), np.std(count_1))
print(np.average(count_2), np.std(count_2))
print(np.average(count_3), np.std(count_3))
print(np.average(count_4), np.std(count_4))
print(np.average(count_5), np.std(count_5))
print(np.average(count_6), np.std(count_6))
print(np.average(count_7), np.std(count_7))
print(np.average(count_8), np.std(count_8))
print(np.average(count_9), np.std(count_9))
print(np.average(count_10), np.std(count_10))
print(np.average(count_11), np.std(count_11))
print(np.average(count_12), np.std(count_12))
print(np.average(count_13), np.std(count_13))









