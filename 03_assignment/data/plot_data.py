'''
Brief: Plot data

Authors: 
         Javier Almario R
         Alvaro Provencio

'''

import numpy as np
import matplotlib.pyplot as plt

# Import data sequential
steps, pi_value, time = np.loadtxt("sequential_mean_pi_timing.csv", comments="#", unpack=True)


time = np.log(time/1000)

steps = np.log(steps)


# Import data parallel
threads, time2 = np.loadtxt("pi_taylor_parallel_threads_means.csv", comments="#", unpack=True)


###################################################
#-------------------Plot sequential.--------------------
plt.title("Execution time: PI Sequential") 
plt.plot(steps,time, label="Pi",marker="8")
plt.xlabel('Number of steps, log scale')
plt.ylabel('Time (s) log scale')
plt.legend(loc="upper right")
plt.grid()
plt.show()

time2 = time2/1000
###################################################
#-------------------Plot parallel.--------------------
plt.title("Execution time: PI Parallized") 
plt.plot(threads,time2, label="Pi",marker="8")
plt.xlabel('Number of Threads')
plt.ylabel('Time (s) log scale')
plt.legend(loc="upper right")
plt.grid()
plt.show()
