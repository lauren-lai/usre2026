import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.optimize import minimize, NonlinearConstraint
import pandas as pd
import csv

def isolate_data(uf_df):
    f_data = np.zeros(len(uf_df))
    uf_df = uf_df[uf_df["SOURCE"].astype(str) == "7"]
    # add timestamp filter
    f_data = uf_df["WND"].str.split(',').str[3].astype(int) / 10
    return f_data

def clean_data(wnd_speeds):
    wnd_speeds[wnd_speeds >= 999.9] = np.nan

    if(np.isnan(wnd_speeds).sum() < (len(wnd_speeds) * 0.05)): #5% sparsity thereshold
        wnd_speeds = wnd_speeds.interpolate(
            method="linear",
            limit_direction="both"
            ).to_numpy()
    else: # if too empty use longest contiguous
        valid = ~np.isnan(wnd_speeds).to_numpy()
        max_len = 0
        start = 0
        current_start = None
        for i in range(len(valid)):
            if valid[i] and current_start is None:
                current_start = i
            elif not valid[i] and current_start is not None:
                length = i - current_start
                if length > max_len:
                    max_len = length
                    start = current_start
                current_start = None

        if current_start is not None:
            length = len(valid) - current_start
            if length > max_len:
                max_len = length
                start = current_start

        wnd_speeds = wnd_speeds[start:start+max_len]

    return wnd_speeds

def show_graph_stats(wnd_speeds, year:str, location:str, bin_size:int):
    bins = np.arange(np.min(wnd_speeds), np.max(wnd_speeds) + bin_size, bin_size)
    plt.hist(wnd_speeds, bins=bins)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")
    plt.title(f"Wind Speeds for {year} in {location}")
    plt.show()

    # plt.scatter(np.arange(len(wnd_speeds)), wnd_speeds, s=10)
    # plt.ylabel("Wind Speed (m/s)")
    # plt.title(f"Wind Speeds for {year} in {location}")
    # plt.xlabel("Hours Over the Year")
    # plt.show()


def get_stats(wnd_speeds, year):
    mean = np.mean(wnd_speeds)
    median = np.median(wnd_speeds)
    std = np.std(wnd_speeds)
    min = np.min(wnd_speeds)
    q1 = np.quantile(wnd_speeds, 0.25)
    q3 = np.quantile(wnd_speeds, 0.75)
    max = np.max(wnd_speeds)
    print(f"Basic Stats for {year}")
    print(f"\tmean={mean}",
        f"\n\tmedian={median}",
        f"\n\tstd={std}",
        f"\n\tmin={min}",
        f"\n\tq1={q1}",
        f"\n\tq3={q3}"
        f"\n\tmax={max}",
        )
    with open("pc-exercise42.txt", "w", encoding="utf-8") as file:
        file.write(f"\nBasic Stats for {year}")
        file.write(
            f"\n\tmean={mean}",
            f"\n\tmedian={median}",
            f"\n\tstd={std}",
            f"\n\tmin={min}",
            f"\n\tq1={q1}",
            f"\n\tq3={q3}"
            f"\n\tmax={max}",
            )
    return [mean, median, std, min, q1, q3, max]

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

bin_size = 1
location = "Seattle"

df_2022 = pd.read_csv("/content/2022.csv")
df_2023 = pd.read_csv("/content/2023.csv")
df_2024 = pd.read_csv("/content/2024.csv")

og_2022 = isolate_data(df_2022)
clean_2022 = clean_data(og_2022)
stats_2022 = get_stats(clean_2022, "2022")

og_2023 = isolate_data(df_2023)
clean_2023 = clean_data(og_2023)
stats_2023 = get_stats(clean_2023, "2023")

og_2024 = isolate_data(df_2024)
clean_2024 = clean_data(og_2024)
stats_2024 = get_stats(clean_2024, "2024")

years = [clean_2022, clean_2023, clean_2024]
stats = [stats_2022, stats_2023, stats_2024]

T = 10
N = 100
dt = T / N
M = 30
num_years = 3

tc_estimates = np.zeros((3, M, num_years))
slsqp_estimates = np.zeros((3, M, num_years))

tc_means = np.zeros((3, num_years))
tc_stds = np.zeros((3, num_years))

slsqp_means = np.zeros((3, num_years))
slsqp_stds = np.zeros((3, num_years))

bounds = [(1e-6, np.inf),(1e-6, np.inf), (1e-2, np.inf)]
feller_cond = lambda x: 2*x[0]*x[1] - x[2]**2
tc_constraint = NonlinearConstraint(feller_cond, 0.0, np.inf)
slsqp_constraint = {'type': 'ineq', 'fun': feller_cond}

for j in range(num_years):
    path = years[j]
    stat = stats[j]
    for i in range(M):
        kappa_guess = np.random.uniform(1e-6, 10)
        theta_guess = np.random.uniform(stat[4], stat[5])
        sigma_guess = np.random.uniform(stat[2]*0.75, stat[2]*1.25)

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

        slsqp_estimates[:, i, j] = slsqp_result.x
        tc_estimates[:, i, j] = tc_result.x

    slsqp_means[:, j] = np.mean(slsqp_estimates[:, :, j], axis=1)
    slsqp_stds[:, j] = np.std(slsqp_estimates[:, :, j], axis=1)

    tc_means[:, j] = np.mean(tc_estimates[:, :, j], axis=1)
    tc_stds[:, j]  = np.std(tc_estimates[:, :, j], axis=1)

    with open("pc-exercise42.txt", "w", encoding="utf-8") as file:
        file.write(f"\nYear {2022+j}")
        file.write("\n\tSLSQP")
        file.write("\n\tmeans")
        file.write(f"\n\tkappa={slsqp_means[0][j]:4f}, theta={slsqp_means[1][j]:4f}, sigma={slsqp_means[2][j]:4f}")
        file.write("\n\tstandard deviations")
        file.write(f"\n\tkappa={slsqp_stds[0][j]:4f}, theta={slsqp_stds[1][j]:4f}, sigma={slsqp_stds[2][j]:4f}")

        file.write("\nn\tTRUST-CONSTR")
        file.write("\n\tmeans")
        file.write(f"\n\tkappa={tc_means[0][j]:4f}, theta={tc_means[1][j]:4f}, sigma={tc_means[2][j]:4f}")
        file.write("\n\tstandard deviations")
        file.write(f"\n\tkappa={tc_stds[0][j]:4f}, theta={tc_stds[1][j]:4f}, sigma={tc_stds[2][j]:4f}")
