import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import log


def calc_weight(database, sample):
    mean_d = np.average(database)
    mean_s = np.average(sample)
    weight = mean_s-mean_d
    return weight

def calc_reward(prob, rand):
    if (rand <= prob):
        return 1
    else:
        return 0
def takeFirst(elem):
    return abs(elem[0])
def find_highest_lines(weight_list, size):
    leaders = []
    cutoff = 0
    index = 0
    for i in weight_list:
        maxx = abs(max(i, key = abs))
        if (maxx >= cutoff):
            if (len(leaders) == size):
                leaders.sort(key = takeFirst)
                leaders.pop(0)
            leaders.append((maxx, index))
            leaders.sort(key = takeFirst)
            cutoff = leaders[0][0]
        index += 1
    return [x[1] for x in leaders]
def plot_function(weight_list):
    dummy_list = list(range(1, len(weight_list) + 1))
    winners = find_highest_lines(weight_list, 8)
    for i in range(len(weight_list)):
        plt.scatter(dummy_list, weight_list[i])
        plt.scatter(dummy_list, weight_list[i], color = 'black')
        if (i in winners):
            plt.plot(dummy_list, weight_list[i], label = 'line' + str(i))
        else:
            plt.plot(dummy_list, weight_list[i], color = 'black')

def simulate(rounds, probs):
    round_counter = 0
    epoch_counter = -1
    weight_list = []
    rewards = []
    for i in range (sum(rounds)+1):
        weight_list.append([1]*(sum(rounds)+1))
    for i in rounds:
        epoch_counter += 1
        iterations = rounds[epoch_counter]
        prob = probs[epoch_counter]

        for j in range(iterations):
            round_counter += 1
            reward = calc_reward(prob, random.random())
            rewards.append(reward)
            if (round_counter >= 4):
                for k in range (min(round_counter-1, 30)):
                    kk = k + 1
                    sample = rewards[-kk:]
                    database = rewards[max(0, round_counter - 30):-kk]
                    weight = calc_weight(database, sample)
                    weight_list[round_counter - k][round_counter] = weight_list[round_counter - k][round_counter-1] + weight * len(database) / max(0.1, np.std(database))
                

    plot_function(weight_list)

simulate([30,30], [0.1,0.9])
plt.legend()
plt.show()









