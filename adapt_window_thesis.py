import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#the goal of this program/file is to create am environment & helper functions in order to adaptively estimate our window size
#that we should consider in order to predict the next arm to be chosen


def filt_0(x):
    if (x[0] == 0):
        return True
    else:
        return False
def filt_1(x):
    if (x[0] == 1):
        return True
    else:
        return False


def prob_and_var(choose_arm_mem, if_success_mem):
    zip_list = zip(choose_arm_mem, if_success_mem)
    zip_list_copy = zip(choose_arm_mem, if_success_mem)
    li_0 = list(filter(filt_0, zip_list))
    li_1 = list(filter(filt_1, zip_list_copy))
    success_0 = 0
    tries_0 = len(li_0)
    for i in li_0:
        if (i[1] == 1):
            success_0 += 1
    success_1 = 0
    tries_1 = len(li_1)
    for i in li_1:
        if (i[1] == 1):
            success_1 += 1
    if (tries_0 > 0):
        prob_0 = success_0 / tries_0
    else:
        prob_0 = 0
    if (tries_1 > 0):
        prob_1 = success_1 / tries_1
    else:
        prob_1 = 0
    return ((prob_0, tries_0),(prob_1, tries_1))


"""helper function to create the list of factors"""
def update_mult_factors(mult_factors, factor):

    if (len(mult_factors) == 0):
        mult_factors.append(factor)
    else:
        mult_factors.append(mult_factors[-1] * factor)

def simulate_single_game(rounds, arm0, arm1, factor, incorrect_weight, wind_options, sample_size):
    choose_arm_mem = []
    if_success_mem = []
    window_size_mem = [0]
    memory_list = []
    win_counter = 0
    window_size = 0
    mult_factors = []
    loop_counter = 0
    global_count = 1
    choose_arm = random.randint(0,1)
    for i in rounds:
        iterations_this_round = rounds[loop_counter]
        curr_arm0 = arm0[loop_counter]
        curr_arm1 = arm1[loop_counter]
        for j in range (iterations_this_round):
            success = True
            rand = random.random()
            if (choose_arm == 1):
                choose_arm_mem.append(1)
                if (rand <= curr_arm1):
                    win_counter += 1
                    if_success_mem.append(1)
                    memory_list.append(-1)
                else:
                    success = False
                    if_success_mem.append(0)
                    memory_list.append(incorrect_weight)
            else:
                choose_arm_mem.append(0)
                if (rand <= curr_arm0):
                    win_counter += 1
                    if_success_mem.append(1)
                    memory_list.append(+1)
                else:
                    success = False
                    if_success_mem.append(0)
                    memory_list.append(-incorrect_weight)
            global_count += 1
            window_size = choose_wind_fun(choose_arm_mem, if_success_mem, wind_options, sample_size, window_size)
            window_size_mem.append(window_size)
            update_mult_factors(mult_factors, factor)
            decision_value = sum([a*b for a,b in zip(memory_list[-window_size:], mult_factors[-window_size:])])
            if (decision_value > 0):
                choose_arm = 0
            elif (decision_value < 0):
                choose_arm = 1
            else:
                choose_arm = random.randint(0,1)
        loop_counter += 1
    window_size_mem.pop(-1)
    plot(rounds, window_size_mem, if_success_mem, choose_arm_mem)
    return win_counter
    
def plot(rounds, window_size_mem, if_success_mem, choose_arm_mem):
    my_colors = {0:'red',1:'green'}
    plt.plot(window_size_mem, color = 'black')
    for i in range(len(window_size_mem)):
        plt.scatter(i, window_size_mem[i], color = my_colors.get(if_success_mem[i]))
        plt.annotate(choose_arm_mem[i], (i, window_size_mem[i]))
    plt.xlabel("Round Number")
    plt.ylabel("Window Size")
    plt.title("Example 4.1 Plot")
    tot = 0
    for i in range(len(rounds)-1):
        tot += rounds[i]
        if (i == 0):
            plt.axvline(tot, 0, max(window_size_mem), label='reversal')
        else:
            plt.axvline(tot, 0, max(window_size_mem))    
    plt.legend()
    plt.tight_layout()

def choose_wind_fun(choose_arm_mem, if_success_mem, wind_options, sample_size, window_size):
    l = len(choose_arm_mem)
    loop_counter = -1
    for i in wind_options:
        loop_counter += 1
        """here we make sure that from having a window 4 we then don't just to 16 in the next round"""
        if (l >= i and ((wind_options[(min(loop_counter+1,len(wind_options)-1))] <= window_size) or (window_size == 4) and (loop_counter == len(wind_options)-1))):

            ((prob_0, tries_0),(prob_1, tries_1)) = prob_and_var(choose_arm_mem[-i:-sample_size], if_success_mem[-i:-sample_size])
            ((prob_0_samp, tries_0_samp),(prob_1_samp, tries_1_samp)) = prob_and_var(choose_arm_mem[-sample_size:], if_success_mem[-sample_size:])

            threshold_0 = 1/sample_size
            threshold_1 = 1/sample_size

            if (abs(prob_0 - prob_0_samp) <= threshold_0 and abs(prob_1 - prob_1_samp) <= threshold_1):
                return i
    return min(l, sample_size)


print(simulate_single_game([30,30,30],[0.7,0.3,0.7],[0.3,0.7,0.3],1.35,1,[16,14,12,10,8],4))
plt.show()


