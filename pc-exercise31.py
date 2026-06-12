import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.optimize import minimize, NonlinearConstraint

def cir_exact_path(kappa, theta, sigma, X0, T, N):
    dt = T / N
    X = np.zeros(N + 1)
    X[0] = X0

    c = (2*kappa) / (sigma**2 * (1 - np.exp(-kappa * dt)))
    d = 4 * kappa * theta / sigma**2
    exp = np.exp(-kappa*dt)

    for n in range(N):
        lam = 2 * c * X[n] * exp
        Y = np.random.noncentral_chisquare(d, lam)
        X[n + 1] = Y / (2 * c)

    return X

def neg_loglike(params, X, dt):

    kappa, theta, sigma = params

    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10

    loglike = 0.0
    c = (2*kappa) / (sigma**2 * (1 - np.exp(-kappa * dt)))
    d = 4 * kappa * theta / sigma**2
    exp = np.exp(-kappa*dt)

    if not np.isfinite(c):
        return 1e10

    if not np.isfinite(d):
        return 1e10

    for i in range(1, len(X)):
        lam = 2 * c * X[i - 1] *exp
        if not np.isfinite(lam):
            return 1e10

        density = 2*c * ncx2.pdf(2*c*X[i], d, lam)

        if density <= 0 or np.isnan(density):
            return 1e10

        loglike += np.log(density)

    return -loglike


kappa = 0.5
theta = 4.0
sigma = 1.0

X0 = 2.0
T = 10
N = 100
dt = T / N
M = 20

tc_estimates = np.zeros((3, M))
slsqp_estimates = np.zeros((3, M))

tc_means = np.zeros(3)
tc_stds = np.zeros(3)

slsqp_means = np.zeros(3)
slsqp_stds = np.zeros(3)

bounds = [(1e-6, np.inf),(1e-6, np.inf), (1e-2, np.inf)]
feller_cond = lambda x: 2*x[0]*x[1] - x[2]**2
tc_constraint = NonlinearConstraint(feller_cond, 0.0, np.inf)
slsqp_constraint = {'type': 'ineq', 'fun': feller_cond}

for i in range(M):
    path = cir_exact_path(kappa, theta, sigma, X0, T, N)
    kappa_guess = np.random.uniform(1e-6, 5)
    theta_guess = np.random.uniform(1e-6, 5)
    sigma_guess = np.random.uniform(1e-6, 5)

    # add tolerances
    slsqp_result = minimize(
        neg_loglike,
        [kappa_guess, theta_guess, sigma_guess],
        args=(path, dt),
        tol=1e-10,
        method="slsqp",
        bounds=bounds,
        constraints=slsqp_constraint
    )

    tc_result = minimize(
        neg_loglike,
        [kappa_guess, theta_guess, sigma_guess],
        args=(path, dt),
        tol=1e-10,
        method="trust-constr",
        bounds=bounds,
        constraints=tc_constraint,
        options={"barrier_tol": 1e-10}
    )

    slsqp_estimates[:, i] = slsqp_result.x
    tc_estimates[:, i] = tc_result.x

    # print(f"slsqp #{i}")
    # print(f"\tsuccess: {slsqp_result.success}")
    # print(f"\tmessage: {slsqp_result.message}")
    # print(f"trust-constr #{i}")
    # print(f"\tsuccess: {tc_result.success}")
    # print(f"\tmessage: {tc_result.message}")
    # print(f"\toptimality: {tc_result.optimality}")


slsqp_means = np.mean(slsqp_estimates, axis=1)
slsqp_stds  = np.std(slsqp_estimates, axis=1)

tc_means = np.mean(tc_estimates, axis=1)
tc_stds  = np.std(tc_estimates, axis=1)

print("SLSQP")
print("means")
print(f"kappa={slsqp_means[0]:4f}, theta={slsqp_means[1]:4f}, sigma={slsqp_means[2]:4f}")
print("standard deviations")
print(f"kappa={slsqp_stds[0]:4f}, theta={slsqp_stds[1]:4f}, sigma={slsqp_stds[2]:4f}")

print("\ntrust-constr")
print("means")
print(f"kappa={tc_means[0]:4f}, theta={tc_means[1]:4f}, sigma={tc_means[2]:4f}")
print("standard deviations")
print(f"kappa={tc_stds[0]:4f}, theta={tc_stds[1]:4f}, sigma={tc_stds[2]:4f}")

with open("pc-exercise31.txt", "w", encoding="utf-8") as file:
    file.write("\nSLSQP")
    file.write("\nmeans")
    file.write(f"\nkappa={slsqp_means[0]:4f}, theta={slsqp_means[1]:4f}, sigma={slsqp_means[2]:4f}")
    file.write("\nstandard deviations")
    file.write(f"\nkappa={slsqp_stds[0]:4f}, theta={slsqp_stds[1]:4f}, sigma={slsqp_stds[2]:4f}")

    file.write("\ntrust-constr")
    file.write("\nmeans")
    file.write(f"\nkappa={tc_means[0]:4f}, theta={tc_means[1]:4f}, sigma={tc_means[2]:4f}")
    file.write("\nstandard deviations")
    file.write(f"\nkappa={tc_stds[0]:4f}, theta={tc_stds[1]:4f}, sigma={tc_stds[2]:4f}")
