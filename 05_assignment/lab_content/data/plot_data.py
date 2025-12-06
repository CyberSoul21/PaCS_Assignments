'''
Brief: Plot data

Authors: 
         Javier Almario R
         Alvaro Provencio

'''

import numpy as np
import matplotlib.pyplot as plt

# Import data sequential
#Size (pixels),Throughput (pixels/s),Kernel throughput (GFLOP/s),Kernel global-mem BW (GB/s)
size_x, throughput, kernel_throughput, kernel_global_memory = np.loadtxt("size_throughput_gflops_kbw.csv", comments="#", unpack=True)


#time = np.log(time/1000)
#regions = np.log(regions)


#time = time/1000
###################################################
#-------------------Plot sequential.--------------------
# plt.title("Kernel throughput and global-mem bandwidth vs image size") 
# plt.plot(size_x,kernel_throughput, label="Kernel throughput (GFLOP/s)",marker="8")
# plt.xlabel('Image size (Megapixels)')
# plt.ylabel('Kernel throughput (GFLOP/s)')
# plt.twinx()
# plt.plot(size_x,kernel_global_memory, label="Kernel global-mem BW (GB/s)",marker="8")
# plt.ylabel('Kernel global-mem BW (GB/s)')
# plt.legend(loc="upper right")
# plt.grid()
# plt.show()



# size_x = size_x / 1e6

# fig, ax1 = plt.subplots()

# # Left y-axis: GFLOP/s
# ax1.plot(size_x, kernel_throughput, marker="o")
# ax1.set_xlabel("Image size (Megapixels)")
# ax1.set_ylabel("Kernel throughput (GFLOP/s)")
# ax1.grid(True)

# # Right y-axis: GB/s
# ax2 = ax1.twinx()
# ax2.plot(size_x, kernel_global_memory, marker="s", linestyle="--")
# ax2.set_ylabel("Kernel global-mem BW (GB/s)")



# plt.title("Kernel throughput and global-mem bandwidth vs image size")
# fig.tight_layout()
# plt.show()

size_x = size_x / 1e6

fig, ax1 = plt.subplots()

# Left y-axis: GFLOP/s (red line)
line1, = ax1.plot(size_x, kernel_throughput,
                  marker="o",
                  linestyle="-",
                  color="red",
                  label="Kernel throughput (GFLOP/s)")
ax1.set_xlabel("Image size (Megapixels)", fontweight="bold")
ax1.set_ylabel("Kernel throughput (GFLOP/s)", fontweight="bold")
ax1.grid(True)

# Right y-axis: GB/s
ax2 = ax1.twinx()
line2, = ax2.plot(size_x, kernel_global_memory,
                  marker="s",
                  linestyle="--",
                  label="Kernel global-mem BW (GB/s)")
ax2.set_ylabel("Kernel global-mem BW (GB/s)", fontweight="bold")

# Combine legends from both axes
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title(
    "Kernel throughput and global-mem bandwidth vs image size",
    fontsize=12,
    fontweight="bold"
)
fig.tight_layout()
plt.show()




plt.figure()
plt.plot(size_x, throughput, marker="o")
plt.xlabel("Image size (Megapixels)")
plt.ylabel("Throughput (pixels/s)")
plt.title("Kernel throughput vs image size (pixels/s)")
plt.grid(True)
plt.tight_layout()
plt.show()