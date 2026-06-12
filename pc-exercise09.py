import numpy as np
import matplotlib.pyplot as plt
import time

n = 50

def generate_random_walk(n:int, distribution: str):
  points = []
  walk = np.zeros(n)
  match distribution:
    case "two point":
      points = np.random.choice([-1, 1], n)
    case "normal":
      points = np.random.normal(0, 1, n)
    case "exponential":
      points = np.random.exponential(1, n) - 1
    case "uniform":
      points = np.random.uniform(-1, 1, n)

  walk = np.cumsum(points)

  return walk

tp_walk = generate_random_walk(n, "two point")
plt.plot(np.arange(n), tp_walk, alpha=0.7, label="two-point")
plt.title("Two-Point Random Walk")
plt.xlabel("n")
plt.ylabel("distribution sum")
# plt.show()
plt.savefig("tp_walk", dpi=300, bbox_inches='tight')

# normal_walk = generate_random_walk(n, "normal")
# plt.plot(np.arange(n), normal_walk, alpha=0.7, label="normal")
# plt.title("Normal Random Walk")
# plt.xlabel("n")
# plt.ylabel("distribution sum")
# plt.legend()
# plt.show()

# exp_walk = generate_random_walk(n, "exponential")
# plt.plot(np.arange(n), exp_walk, alpha=0.7, label="exponential")
# plt.title("Exponential Random Walk")
# plt.xlabel("n")
# plt.ylabel("distribution sum")
# plt.legend()
# plt.show()

# uni_walk = generate_random_walk(n, "uniform")
# plt.plot(np.arange(n), uni_walk, alpha=0.7, label="uniform")
# plt.title("Uniform Random Walk")
# plt.xlabel("n")
# plt.ylabel("distribution sum")
# plt.legend()
# plt.show()

