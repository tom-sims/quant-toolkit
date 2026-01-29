# === MONTE CARLO OPTION PRICING INPUTS ===
# S0 : spot price now
# K  : strike
# T  : time to maturity in years
# r  : continuously-compounded risk-free rate (annual)
# sigma : volatility (annual)
# q  : continuous dividend yield (annual). default 0.0
#
# n_paths : number of simulated paths (e.g. 50_000 - 500_000)
# n_steps : number of time steps in each path (e.g. 252 for daily steps over 1 year)
#
# option_type : "call" or "put"
#
# antithetic : if True, uses antithetic variates (reduces variance)
# seed : optional random seed for reproducibility
#
# How to call:
#
# price = monte_carlo_price(
#     S0=100, K=105, T=1.0, r=0.03, sigma=0.2,
#     q=0.0, n_paths=100000, n_steps=252, option_type="call",
#     antithetic=True, seed=42
# )
#
# Returns:
# price -> float

import numpy as np

def _validate_inputs(S0, K, T, r, sigma, q, n_paths, n_steps, option_type):
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if int(n_paths) < 1:
        raise ValueError("n_paths must be >= 1.")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1.")

    option_type = str(option_type).lower().strip()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    return option_type


def _payoff(ST, K, option_type):
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def monte_carlo_price(
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    n_paths=100000,
    n_steps=252,
    option_type="call",
    antithetic=True,
    seed=None,
):
    option_type = _validate_inputs(S0, K, T, r, sigma, q, n_paths, n_steps, option_type)

    n_paths = int(n_paths)
    n_steps = int(n_steps)

    if seed is not None:
        np.random.seed(int(seed))

    dt = T / float(n_steps)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * T)

    if antithetic:
        half = (n_paths + 1) // 2
        Z = np.random.normal(0.0, 1.0, size=(half, n_steps))
        Z = np.vstack([Z, -Z])
        Z = Z[:n_paths]
    else:
        Z = np.random.normal(0.0, 1.0, size=(n_paths, n_steps))

    log_paths = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    ST = np.exp(log_paths[:, -1])

    payoff = _payoff(ST, K, option_type)
    price = disc * payoff.mean()
    return float(price)


def monte_carlo_price_with_ci(
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    n_paths=100000,
    n_steps=252,
    option_type="call",
    antithetic=True,
    seed=None,
    ci=0.95,
):
    option_type = _validate_inputs(S0, K, T, r, sigma, q, n_paths, n_steps, option_type)

    n_paths = int(n_paths)
    n_steps = int(n_steps)

    if seed is not None:
        np.random.seed(int(seed))

    dt = T / float(n_steps)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * T)

    if antithetic:
        half = (n_paths + 1) // 2
        Z = np.random.normal(0.0, 1.0, size=(half, n_steps))
        Z = np.vstack([Z, -Z])
        Z = Z[:n_paths]
    else:
        Z = np.random.normal(0.0, 1.0, size=(n_paths, n_steps))

    log_paths = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    ST = np.exp(log_paths[:, -1])

    payoff = _payoff(ST, K, option_type)
    discounted = disc * payoff

    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(n_paths))

    # Normal approx CI
    # 95% -> z ~ 1.96, 90% -> 1.645, 99% -> 2.576
    ci = float(ci)
    if ci <= 0 or ci >= 1:
        raise ValueError("ci must be between 0 and 1 (e.g. 0.95).")

    z_map = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    z = z_map.get(round(ci, 2), 1.959964)

    lo = price - z * se
    hi = price + z * se

    return {
        "price": price,
        "stderr": se,
        "ci": ci,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
