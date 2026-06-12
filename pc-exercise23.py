import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

kappa = 0.5
theta = 4.0
sigma = 1.5
x = 2.0
T = [0.01, 0.1, 1, 10]
y = np.linspace(0, 20, 10000)

for t in T:
    c = 2*kappa / (sigma**2 * (1 - np.exp(-kappa*t)))
    d = 4*kappa*theta / sigma**2
    lam = 2*c*x*np.exp(-kappa*t)

    density = 2*c * ncx2.pdf(2*c*y, df=d, nc=lam)

    plt.plot(y, density, label=f"Δt={t}")

plt.xlabel("Future Value Y")
plt.ylabel("Density")
plt.title(f"CIR Transition Density")
plt.legend()
plt.savefig("cir_transition_density", dpi=300, bbox_inches='tight')