import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

"""
LIST
- better exponential
    think about what would work
    simulate all possible options for 50 rounds to get best parameters (how many to consider, which weights, which positiv or negative impact)
    more from a logical/intuitional perspective first

- hypothesis test style
    look into how we can learn a distribution and how we can then test a result on that hypothesis
    create a simple program

- take into account variance
- look at random probability distributions
"""


"""helper function to create the list of factors"""
def createfactors(window_size, factor):
    li = []
    for i in range(window_size):
        li.append(pow(factor,i+1))
    return li



"""
exp_imp code from fast_learning_examples.py
"""

def exp_imp(rounds, arm0, arm1):
    win_counter = 0
    choose_arm = random.randint(0,1)
    memory_list = [0] * 4
    mult_factors = createfactors(4, 1.3)
    loop_counter = 0
    for i in rounds:
        iterations_this_round = rounds[loop_counter]
        curr_arm0 = arm0[loop_counter]
        curr_arm1 = arm1[loop_counter]
        for j in range (iterations_this_round):
            rand = random.random()
            memory_list.pop(0)
            if (choose_arm == 1):
                if (rand <= curr_arm1):
                    win_counter += 1
                    memory_list.append(-1)
                else:
                    memory_list.append(+1.3)
            else:
                if (rand <= curr_arm0):
                    win_counter += 1
                    memory_list.append(+1)
                else:
                    memory_list.append(-1.3)

            decision_value = sum([a*b for a,b in zip(memory_list, mult_factors)])
            if (decision_value > 0):
                choose_arm = 0
            elif (decision_value < 0):
                choose_arm = 1
            else:
                choose_arm = random.randint(0,1)
        loop_counter += 1
    return win_counter

def test_exp_imp(num_tests, num_rounds, arm0, arm1):
    exp_imp_count = 0
    list_for_plot = []
    for i in range (num_tests):
        exp_imp_count += exp_imp(num_rounds, arm0, arm1)
        list_for_plot.append(exp_imp_count/((i+1)*sum(num_rounds)))

    plt.plot(list(range(0,num_tests,1)),list_for_plot, 'b')
    plt.xlabel("Number of Simulations")
    plt.ylabel("Average Winning Probability")
    plt.title("Algorithm 'exponential_importance'")
    plt.show()

    exp_imp_avg = exp_imp_count / num_tests
    print("--------------------------------------------TEST------------------------------------------------")
    print("Average wins for exponential_importance with reversals is:",  exp_imp_avg)
    print("--------------------------------------------------------------------------------------------------")

"""test_exp_imp(100000, [11,12,7], [0.8,0.2,0.8], [0.2,0.8,0.2])"""
"""test_exp_imp(100000, [9,7,14], [0.9,0.4,0.8], [0.1,0.6,0.2])"""


"""helper function to calculate the optimal expected value"""
def optimal_value(rounds, arm0, arm1):
    counter = 0
    loop_counter = 0
    for i in rounds:
        iterations_this_round = rounds[loop_counter]
        curr_arm0 = arm0[loop_counter]
        curr_arm1 = arm1[loop_counter]
        counter += iterations_this_round * max(curr_arm0, curr_arm1)
        loop_counter += 1
    return counter


"""this algorithm is for a basis to simulate the different parameters in order to find an optimal choice"""
""" we don't have to have the correct_weight parameter as only the relation of it to the incorrect_weight and factor counts"""
def exp_imp_sim(iterations, rounds, arm0, arm1, window_size, factor, incorrect_weight):
    result_list = []
    for l in range(iterations):
        win_counter = 0
        choose_arm = random.randint(0,1)
        memory_list = [0] * window_size
        mult_factors = createfactors(window_size, factor)
        loop_counter = 0
        for i in rounds:
            iterations_this_round = rounds[loop_counter]
            curr_arm0 = arm0[loop_counter]
            curr_arm1 = arm1[loop_counter]
            for j in range (iterations_this_round):
                rand = random.random()
                memory_list.pop(0)
                if (choose_arm == 1):
                    if (rand <= curr_arm1):
                        win_counter += 1
                        memory_list.append(-1)
                    else:
                        memory_list.append(incorrect_weight)
                else:
                    if (rand <= curr_arm0):
                        win_counter += 1
                        memory_list.append(+1)
                    else:
                        memory_list.append(-incorrect_weight)

                decision_value = sum([a*b for a,b in zip(memory_list, mult_factors)])
                if (decision_value > 0):
                    choose_arm = 0
                elif (decision_value < 0):
                    choose_arm = 1
                else:
                    choose_arm = random.randint(0,1)
            loop_counter += 1
        result_list.append((win_counter / sum(rounds)))
    return result_list


def para_simulator(iterations, rounds, arm0, arm1):
    all_results = []
    opt = optimal_value(rounds, arm0, arm1) / sum(rounds)
    for window_size in range(1,10,1):
        """factor below 1 doesn't make sense and above 2 will give too much weight to last result"""
        for s_factor in range(10,20,1):
            factor = s_factor / 10.
            for s_incorrect_weight in range (1,15,1):
                incorrect_weight = s_incorrect_weight / 10.
                result_list = exp_imp_sim(iterations, rounds, arm0, arm1, window_size, factor, incorrect_weight)
                var = np.var(result_list)
                med = np.median(result_list)
                av = np.average(result_list)
                all_results.append((window_size, factor, incorrect_weight, av, var, med))
    sorted_by_av_rev = sorted(all_results, key=lambda tup: tup[3], reverse = True)
    sorted_by_var = sorted(all_results, key=lambda tup: tup[4])
    return (opt, sorted_by_av_rev, sorted_by_var)
    
def output_simulator_results(iterations, rounds, arm0, arm1):
    (opt, sorted_by_av_rev, sorted_by_var) = para_simulator(iterations, rounds, arm0, arm1)
    print("-----------------------RESULTS OF SIMULATION---------------------------")
    print("")
    print ("The optimal value for the average (target) is: ", opt)
    print("")
    print("\033[1m" + "The following 3 parameterizations maximized the average:" + "\033[0m")
    print("Nr. 1:   " + "Window size: ", sorted_by_av_rev[0][0], " Factor: ", sorted_by_av_rev[0][1], " Weight for error: ", sorted_by_av_rev[0][2], " Average: ", sorted_by_av_rev[0][3], " Variance: ", sorted_by_av_rev[0][4], " Median: ", sorted_by_av_rev[0][5])
    print("Nr. 2:   " + "Window size: ", sorted_by_av_rev[1][0], " Factor: ", sorted_by_av_rev[1][1], " Weight for error: ", sorted_by_av_rev[1][2], " Average: ", sorted_by_av_rev[1][3], " Variance: ", sorted_by_av_rev[1][4], " Median: ", sorted_by_av_rev[1][5])
    print("Nr. 3:   " + "Window size: ", sorted_by_av_rev[2][0], " Factor: ", sorted_by_av_rev[2][1], " Weight for error: ", sorted_by_av_rev[2][2], " Average: ", sorted_by_av_rev[2][3], " Variance: ", sorted_by_av_rev[2][4], " Median: ", sorted_by_av_rev[2][5])
    print("Nr. 4:   " + "Window size: ", sorted_by_av_rev[3][0], " Factor: ", sorted_by_av_rev[3][1], " Weight for error: ", sorted_by_av_rev[3][2], " Average: ", sorted_by_av_rev[3][3], " Variance: ", sorted_by_av_rev[3][4], " Median: ", sorted_by_av_rev[3][5])
    print("Nr. 5:   " + "Window size: ", sorted_by_av_rev[4][0], " Factor: ", sorted_by_av_rev[4][1], " Weight for error: ", sorted_by_av_rev[4][2], " Average: ", sorted_by_av_rev[4][3], " Variance: ", sorted_by_av_rev[4][4], " Median: ", sorted_by_av_rev[4][5])
    print("")
    print("\033[1m" + "The following 3 parameterizations minimized the variance:" + "\033[0m")
    print("Nr. 1:   " + "Window size: ", sorted_by_var[0][0], " Factor: ", sorted_by_var[0][1], " Weight for error: ", sorted_by_var[0][2], " Average: ", sorted_by_var[0][3], " Variance: ", sorted_by_var[0][4], " Median: ", sorted_by_var[0][5])
    print("Nr. 2:   " + "Window size: ", sorted_by_var[1][0], " Factor: ", sorted_by_var[1][1], " Weight for error: ", sorted_by_var[1][2], " Average: ", sorted_by_var[1][3], " Variance: ", sorted_by_var[1][4], " Median: ", sorted_by_var[1][5])
    print("Nr. 3:   " + "Window size: ", sorted_by_var[2][0], " Factor: ", sorted_by_var[2][1], " Weight for error: ", sorted_by_var[2][2], " Average: ", sorted_by_var[2][3], " Variance: ", sorted_by_var[2][4], " Median: ", sorted_by_var[2][5])
    print("Nr. 4:   " + "Window size: ", sorted_by_var[3][0], " Factor: ", sorted_by_var[3][1], " Weight for error: ", sorted_by_var[3][2], " Average: ", sorted_by_var[3][3], " Variance: ", sorted_by_var[3][4], " Median: ", sorted_by_var[3][5])
    print("Nr. 5:   " + "Window size: ", sorted_by_var[4][0], " Factor: ", sorted_by_var[4][1], " Weight for error: ", sorted_by_var[4][2], " Average: ", sorted_by_var[4][3], " Variance: ", sorted_by_var[4][4], " Median: ", sorted_by_var[4][5])

"""output_simulator_results(10, [9,7,14], [0.9,0.4,0.8], [0.1,0.6,0.2])"""

"""helper function to calculate random round values for a random game satisfying min_distance"""
def calc_round(totalrounds, min_distance, reversals):
    revs = []
    rounds = []
    for i in range(reversals):
        revs.append(random.randint(min_distance,totalrounds-min_distance))
    revs.sort()
    problem = False
    rounds.append(revs[0])
    for i in range(1, len(revs), 1):
        if (revs[i]-revs[i-1] < min_distance):
            problem = True
        else:
            rounds.append(revs[i]-revs[i-1])
    rounds.append(totalrounds - revs[len(revs)-1])
    if (problem == True):
        return calc_round(totalrounds, min_distance, reversals)
    else:
        return rounds

"""print(calc_round(30, 5, 2))"""

"""helper function to sample random probabilities from a list of options whereby the sum of both arms = 1"""
def calc_probs_simple(probs_list, reversals):
    arm0 = []
    arm1 = []
    for i in range(reversals+1):
        sample = random.randint(0, len(probs_list) - 1)
        arm0.append(probs_list[sample])
        arm1.append(1 - probs_list[sample])
    return (arm0, arm1)

"""helper function to sample random probabilities from a lost of options whereby the two arms are independant"""
def calc_probs_complex(probs_list, reversals):
    arm0 = []
    arm1 = []
    for i in range(reversals+1):
        sample_1 = random.randint(0, len(probs_list) - 1)
        sample_2 = random.randint(0, len(probs_list) - 1)
        arm0.append(probs_list[sample_1])
        arm1.append(probs_list[sample_2])
    return (arm0, arm1)


"""
this part is used for the histograms 

"""

"""this is the function where we create a random game and then simulate it with various different strategies and for every game save the 10 best strategies"""
def game_simulator(iterations, totalrounds, min_distance, reversals, probs_list):
    all_best_ten = []
    all_windows = []
    all_factors = []
    all_weights = []
    for i in range(30):
        (arm0, arm1) = calc_probs_complex(probs_list, reversals)
        rounds = calc_round(totalrounds, min_distance, reversals)
        opt, sorted_by_av_rev, sorted_by_var = para_simulator(iterations, rounds, arm0, arm1)
        best_ten = sorted_by_av_rev[:10]
        all_windows += list(map(lambda x: x[0], best_ten))
        all_factors += list(map(lambda x: x[1], best_ten))
        all_weights += list(map(lambda x: x[2], best_ten))
        best_ten = list(map(lambda x: x + (opt, x[3]/opt, rounds, arm0, arm1), best_ten))
        all_best_ten = all_best_ten + best_ten
    return (all_best_ten, all_windows, all_factors, all_weights)


"""
this part is used for the histograms and the analysis
"""

def histo(all_windows, all_factors, all_weights):
    print("avg_window: ", np.average(all_windows))
    print("avg_factors: ", np.average(all_factors))
    print("avg_weights: ", np.average(all_weights))
    fig, axs = plt.subplots(1, 3, tight_layout=True)
    axs[0].hist(all_windows)
    axs[0].set(xlabel = 'Window Size')
    axs[1].hist(all_factors)
    axs[1].set(xlabel = 'Decay Factor')
    axs[2].hist(all_weights)
    axs[2].set(xlabel = 'Weights for Wrong')
    plt.show()




(all_best_ten, all_windows, all_factors, all_weights) = game_simulator(100, 90, 10, 2, [0.5, 0.7,0.9])
"""print(list(map(lambda x: (x[0],x[1],x[2],round(x[7],2),x[8],x[9],x[10]),all_best_ten)))"""

"""output format: Window, Factor, WeightforError, %to average / 100,   """

histo(all_windows, all_factors, all_weights)
