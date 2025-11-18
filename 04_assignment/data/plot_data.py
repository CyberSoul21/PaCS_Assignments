'''
Brief: Plot data

Authors: 
         Javier Almario R
         Alvaro Provencio

'''

import numpy as np
import matplotlib.pyplot as plt

# Import data sequential
regions, time = np.loadtxt("regions_vs_time.csv", comments="#", unpack=True)


#time = np.log(time/1000)
#regions = np.log(regions)


time = time/1000
###################################################
#-------------------Plot sequential.--------------------
plt.title("Execution time: Ray tracer") 
plt.plot(regions,time, label="Threads: 8",marker="8")
#plt.xlabel('Number of stepsregions, log scale')
#plt.ylabel('Time (s) log scale')
plt.xlabel('Number of Regions')
plt.ylabel('Time (s)')
plt.legend(loc="upper right")
plt.grid()
plt.show()


