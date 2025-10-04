'''
Brief: Plot data

Authors: 
         Javier Almario R
         Alvaro Provencio

'''

import numpy as np
import matplotlib.pyplot as plt

# Import data
matrix_real_x, matrix_real_y = np.loadtxt("real_matrix.txt", comments="#", unpack=True)
eigen_real_x, eigen_real_y = np.loadtxt("real_eigen.txt", comments="#", unpack=True)

matrix_user_x, matrix_user_y = np.loadtxt("user_matrix.txt", comments="#", unpack=True)
eigen_user_x, eigen_user_y = np.loadtxt("user_eigen.txt", comments="#", unpack=True)

matrix_sys_x, matrix_sys_y = np.loadtxt("sys_matrix.txt", comments="#", unpack=True)
eigen_sys_x, eigen_sys_y = np.loadtxt("sys_eigen.txt", comments="#", unpack=True)




###################################################
#-------------------Plot real.--------------------
plt.title("Execution time: Real") 
plt.plot(matrix_real_x, matrix_real_y, label="Matrix normal",marker="8")
plt.plot(eigen_real_x, eigen_real_y, label="Matrix with Eigen",marker="8")
plt.xlabel('Matrix dimension (nxn)')
plt.ylabel('Time (seconds)')
plt.legend(loc="upper right")
plt.grid()
plt.show()

###################################################
#-------------------Plot user.--------------------
plt.title("Execution time: user") 
plt.plot(matrix_user_x, matrix_user_y, label="Matrix normal",marker="8")
plt.plot(eigen_user_x, eigen_user_y, label="Matrix with Eigen",marker="8")
plt.xlabel('Matrix dimension (nxn)')
plt.ylabel('Time (seconds)')
plt.legend(loc="upper right")
plt.grid()
plt.show()

###################################################
#-------------------Plot sys.--------------------
plt.title("Execution time: sys") 
plt.plot(matrix_sys_x, matrix_sys_y, label="Matrix normal",marker="8")
plt.plot(eigen_sys_x, eigen_sys_y, label="Matrix with Eigen",marker="8")
plt.xlabel('Matrix dimension (nxn)')
plt.ylabel('Time (seconds)')
plt.legend(loc="upper right")
plt.grid()
plt.show()