import numpy as np
import matplotlib.pyplot as plt

# PART A
def brownian_path(total_time, steps):
    delta_t = total_time/steps
    sqrt_delta_t = np.sqrt(delta_t)
    t_vector = np.linspace(0, total_time, steps+1)

    z_vector = np.random.standard_normal(steps)
    WP_vector = np.zeros(steps+1)

    for i in range(steps):
        WP_vector[i+1] = WP_vector[i] + sqrt_delta_t*z_vector[i]

    return t_vector, WP_vector

# PART B
b_T = 1
b_N = [10, 100, 1000, 10000]

for i in range(len(b_N)):
    tn, w = brownian_path(b_T, b_N[i])
    plt.subplot(4, 1, i+1)
    plt.plot(tn, w)
plt.xlabel("tn")
plt.ylabel("W")
plt.suptitle("Brownian Paths: Same T + Different N")
plt.show()

# PART C
c_T = 1
c_N = 1000
for i in range(10):
    tn, w = brownian_path(c_T, c_N)
    plt.plot(tn, w)
plt.xlabel("tn")
plt.ylabel("W")
plt.title("Brownian Paths")
plt.savefig("brownian_paths", dpi=300, bbox_inches='tight')


# # PART D + E
# M = 10**4
# end_points = np.zeros(M)
# mid_points = np.zeros(M)
# d_T = 1
# d_N = 100

# for i in range(M):
#     tn, w = brownian_path(d_T, d_N)
#     end_points[i] = w[-1]
#     mid_points[i] = w[int(d_N/2)]

# bin_size_end = 1e-1
# bins_end = np.arange(min(end_points), max(end_points) + bin_size_end, bin_size_end)
# plt.hist(end_points, bins=bins_end)
# plt.title("End Points of Brownian Paths")
# plt.xlabel("End Point")
# plt.ylabel("Frequency")
# plt.show()

# bid_size_mid = 1e-1
# bins_mid = np.arange(min(mid_points), max(mid_points) + bid_size_mid, bid_size_mid)
# plt.hist(mid_points, bins=bins_mid)
# plt.title("Mid Points of Brownian Paths")
# plt.xlabel("Mid Point")
# plt.ylabel("Frequency")
# plt.show()