import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import log
import itertools

#random.seed(1456789)

def calc_weight(database, sample):
    mean_d = np.average(database)
    mean_s = np.average(sample)
    weight = mean_s-mean_d
    return weight

def calc_reward(arm_rewards, rand):
    reward = 0
    for (pot, prob) in arm_rewards:
        if (rand <= prob):
            reward += pot
    return reward

def calc_rewards_for_arm(start_for_arm, rewards_for_arm):
    rews = []
    for i in rewards_for_arm:
        if (i[1] >= start_for_arm):
            rews.append(i)
    return rews 

def calc_rewards_for_arm_2(start_for_arm, rewards_for_arm):
    rews = []
    for i in rewards_for_arm:
        if (i[1] >= start_for_arm):
            rews.append(i[0])
    return rews 

def decide_for_arm_max_nd(arms, norm_parameters):
    list_of_max_arm = []
    max_norm_value = -1000
    for i in range(arms):
        norm_value = random.normalvariate(norm_parameters[i][0], norm_parameters[i][1])
        if (norm_value == max_norm_value):
            list_of_max_arm.append(i)
        if (norm_value > max_norm_value):
            max_norm_value = norm_value
            list_of_max_arm = [i]
    return random.choice(list_of_max_arm)

def calculate_norm_parameters(weight_list, round_counter, starts_of_database, rewards, choose_arm, old_norm_para_list):
    norm_para_list = []
    for i in range(len(weight_list)):
        if (i == choose_arm):
            mu = 0
            sigma = 0
            total_alpha_hat = 0
            total_alpha = 0
            for j in range(len(weight_list[i][0][starts_of_database[i]:round_counter+1])):
                if ((j+starts_of_database[i]) in [b for (c,b) in rewards[i]]):
                    total_alpha += np.exp(weight_list[i][0][j+starts_of_database[i]][round_counter])
            if (total_alpha == 0):
                norm_para_list.append([0.5, 0.25])
                continue
            for j in range(len(weight_list[i][0][starts_of_database[i]:round_counter+1])):
                if ((j+starts_of_database[i]) in [b for (c,b) in rewards[i]]):
                    alpha = np.exp(weight_list[i][0][j+starts_of_database[i]][round_counter])
                    alpha_hat = alpha / total_alpha
                    total_alpha_hat += alpha_hat
                    if (len(calc_rewards_for_arm(j+starts_of_database[i], rewards[i])) == 0):
                        mu += alpha_hat * 0.5
                        sigma += alpha_hat * 0.1
                    else:
                        mu += alpha_hat * np.average(calc_rewards_for_arm_2(j+starts_of_database[i], rewards[i]))
                        sigma += alpha_hat * np.std(calc_rewards_for_arm_2(j+starts_of_database[i], rewards[i]))
        if (i != choose_arm):
            mu = old_norm_para_list[i][0] + 0.1 * (0.5-old_norm_para_list[i][0])
            sigma = old_norm_para_list[i][1] + 0.1 * (0.25-old_norm_para_list[i][1])
        norm_para_list.append([mu, sigma])
    return norm_para_list

def simulate(rounds, probs):
    num_arms = len(probs)
    choose_arm = random.randint(0,num_arms-1)
    round_counter = 0
    epoch_counter = -1
    total_reward = 0
    rewards_to_consider = []
    weight_list = []
    rewards = []
    starts_of_database = []
    norm_parameters = []
    for i in range(num_arms):
        norm_parameters.append([0.5,0.25])
        weight_list.append([[],[]])
        rewards.append([(0,round_counter)])
        starts_of_database.append(1)
        rewards_to_consider.append([])
        for ii in range (sum(rounds)+1):
            weight_list[i][0].append([0]*(sum(rounds)+1))
            weight_list[i][1].append([0]*(sum(rounds)+1))
    for i in rounds:
        epoch_counter += 1
        iterations = rounds[epoch_counter]
        probs_this_epoch = []
        for j in range(num_arms):
            probs_this_epoch.append(probs[j][epoch_counter])
        for j in range(iterations):
            round_counter += 1
            reward = calc_reward(probs_this_epoch[choose_arm], random.random())
            total_reward += reward
            rewards[choose_arm].append((reward, round_counter))
            for m in range(num_arms):
                if (m != choose_arm):
                    for n in range(round_counter):
                        weight_list[m][0][n+1][round_counter] = 0.9 * weight_list[m][0][n+1][round_counter - 1]
                        weight_list[m][1][n+1][round_counter] = 0.9 * weight_list[m][1][n+1][round_counter - 1]
                else:
                    max_weight = 0
                    max_round = 1
                    rewards_for_arm = calc_rewards_for_arm(starts_of_database[m], rewards[m])
                    if (len(rewards_for_arm) >= 2):
                        for k in range (len(rewards_for_arm)-1):
                            kk = k + 1
                            sample = [a for (a,b) in rewards[m][-kk:]]
                            database = [a for (a,b) in rewards[m][-len(rewards_for_arm):-kk]]
                            weight = calc_weight(database, sample)
                            potential_reversal = rewards_for_arm[-kk][1]
                            weight_list[m][0][potential_reversal][round_counter] = weight_list[m][0][potential_reversal][round_counter-1] + weight * len(database) / (10 * max(0.1, np.std(database)))
                            weight_list[m][1][potential_reversal][round_counter] = weight_list[m][1][potential_reversal][round_counter-1] - weight * len(database) / (10 * max(0.1, np.std(database)))
                            if (weight_list[m][0][potential_reversal][round_counter] > max_weight):
                                max_weight = weight_list[m][0][potential_reversal][round_counter]
                                max_round = potential_reversal
                            if (weight_list[m][1][potential_reversal][round_counter] > max_weight):
                                max_weight = weight_list[m][1][potential_reversal][round_counter]
                                max_round = potential_reversal   
                        if (max_weight >= 3):
                            starts_of_database[m] = max_round
                            new_weight_list = [[],[]]
                            for i in range (sum(rounds)+1):
                                new_weight_list[0].append([0]*(sum(rounds)+1))
                                new_weight_list[1].append([0]*(sum(rounds)+1))
                            reset_weights(starts_of_database[m], calc_rewards_for_arm(starts_of_database[m], rewards[m]), round_counter, new_weight_list)
                            weight_list[m] = new_weight_list
            print("For round: ", round_counter, "we chose arm: ", choose_arm, "we got reward: ", reward, "and have a total reward of: ",  total_reward)
            print("Norm_dist parameters for am 0: ", norm_parameters[0], "Norm_dist parameters for am 1: ", norm_parameters[1])
            print("Last 'confirmed' reversal for arm 0: ", starts_of_database[0], "Last 'confirmed' reversal for arm 1: ", starts_of_database[1])
            print("----------------------------------------------------------------------------------------------------------------------------------")
            norm_parameters = calculate_norm_parameters(weight_list, round_counter, starts_of_database, rewards, choose_arm, norm_parameters)
            choose_arm = decide_for_arm_max_nd(num_arms, norm_parameters)
    return total_reward

def reset_weights(start_of_database, new_rewards, round_counter, new_weight_list):
    fake_round_counter = start_of_database - 1
    for i in range (round_counter - start_of_database + 1):
        fake_round_counter += 1
        rounds = fake_round_counter - start_of_database + 1
        real_rounds = [b for (a,b) in new_rewards if b <= fake_round_counter]
        len_real_rounds = len(real_rounds)
        real_new_rewards = [(a,b) for (a,b) in new_rewards if b <= fake_round_counter]
        if ((fake_round_counter in [b for (a,b) in new_rewards]) and len_real_rounds >= 2):
            for k in range(len_real_rounds-1):
                kk = k+1
                sample = [a for (a,b) in real_new_rewards[-kk:]]
                database = [a for (a,b) in real_new_rewards[:-kk]]
                weight = calc_weight(database, sample)
                new_weight_list[0][real_rounds[-kk]][fake_round_counter] = new_weight_list[0][real_rounds[-kk]][fake_round_counter-1] + weight * len(database) / (10 * max(0.1, np.std(database)))
                new_weight_list[1][real_rounds[-kk]][fake_round_counter] = new_weight_list[1][real_rounds[-kk]][fake_round_counter-1] - weight * len(database) / (10 * max(0.1, np.std(database)))
        else:
            for k in range(rounds-1):
                new_weight_list[0][fake_round_counter - k][fake_round_counter] = 0.9 * new_weight_list[0][fake_round_counter - k][fake_round_counter-1]
                new_weight_list[1][fake_round_counter - k][fake_round_counter] = 0.9 * new_weight_list[1][fake_round_counter - k][fake_round_counter-1]




#print(simulate([30,30,30], [[[(1,0.9)],[(1,0.1)],[(1,0.9)]],[[(1,0.1)],[(1,0.9)],[(1,0.1)]]]))

count = 0
for i in range(100):
    count += simulate([20,20,20], [[[(1,0.9)],[(1,0.1)],[(1,0.9)]],[[(1,0.1)],[(1,0.9)],[(1,0.1)]]])
print(count / 100)





