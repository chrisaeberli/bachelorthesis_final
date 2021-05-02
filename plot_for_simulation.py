import random 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#for first simulation

"""avg_sim_1 = [57.29, 57.25, 57.3, 57.29, 57.32, 57.31, 57.01, 56.91, 56.97, 53.78, 53.87, 53.87, 53.5, 53.5, 53.93, 53.48, 53.74, 53.63, 46.26, 46.15, 46.67, 47.44, 47.16, 47.36, 47.31, 47.59, 47.71, 46.31, 46.38, 46.64, 47.44, 47.21, 47.9, 47.71, 48.42, 47.68]
avg_sim_2 = [47.27, 46.6, 46.74, 47.34, 47.01, 47.42, 46.83, 47.75, 47.28, 40.92, 41.62, 41.38, 41.71, 42.12, 42.53, 43, 42.41, 41.94, 40.61, 40.08, 40.22, 40.22, 40.04, 40.75, 40.81, 40.88, 40.19, 39.95, 40.48, 39.9, 40.31, 40.58, 40.09, 40.66, 41.55, 40.42]
avg_sim_3 = [33.87, 33.82, 34.07, 34.51, 34.33, 33.74, 34.25, 34.09, 33.65, 33.34, 33.7, 32.73, 32.8, 32.87, 33.01, 31.78, 32.83, 33.06, 32.53, 32.25, 32.55, 32.4, 32.91, 32.21, 32.69, 32.17, 32.55, 32.89, 32.65, 32.45, 32.59, 32.16, 32.46, 32.5, 32.67, 33.1]
avg_sim_4 = [50.54, 51.99, 52.47, 51.08, 50.23, 51.82, 51.06, 50.58, 51.55, 44.65, 43.94, 44.11, 44.79, 43.52, 44.02, 43.99, 44.6, 43.17, 41.75, 41.7, 43.07, 44.39, 43.66, 41.96, 42.95, 42.44, 41.84, 41.88, 41.6, 42.23, 42.94, 42.91, 41.84, 41.43, 42.15, 41.35]
plt.plot(avg_sim_1, label = 'Scenario 1')
plt.plot(avg_sim_2, label = 'Scenario 2')
plt.plot(avg_sim_3, label = 'Scenario 3')
plt.plot(avg_sim_4, label = 'Scenario 4')
for i in [9,18,27]:
    if (i == 9):
        plt.axvline(i, label = 'Change in scaling parameter', color = 'black', linestyle = '--')
    else:
        plt.axvline(i, color = 'black', linestyle = '--')
       
plt.xlabel("Parameter Combination")
plt.ylabel("Average Total Reward")
plt.title("Parameter Simulation Analysis")
plt.legend()
plt.tight_layout()
plt.show()"""


#for second simulation

avg_sim_1_2 = [57.33, 57.28, 57.25, 57.18, 57.26, 57.32]
avg_sim_2_2 = [47.3, 46.93, 47.05, 46.14, 46.61, 44.63]
avg_sim_3_2 = [34.55, 33.8, 34.77, 34.39, 33.68, 32.82]
avg_sim_4_2 = [51.88, 52.18, 52.39, 53.29, 51.79, 48.82]
x = [5, 7.5, 10, 15, 25, 50]
xx = [str(va) for va in x]
plt.plot(xx, avg_sim_1_2, label = 'Scenario 1')
plt.plot(xx, avg_sim_2_2, label = 'Scenario 2')
plt.plot(xx, avg_sim_3_2, label = 'Scenario 3')
plt.plot(xx, avg_sim_4_2, label = 'Scenario 4')
plt.xlabel("Scaling Paramater s")
plt.ylabel("Average Total Reward")
plt.title("Parameter Simulation Analysis 2")
plt.legend()
plt.tight_layout()
plt.show()



