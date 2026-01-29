# === BLACK-SCHOLES INPUTS ===
# S0 : spot price now
# K  : strike
# T  : time to maturity in years
# r  : continuously-compounded risk-free rate (annual)
# sigma : volatility (annual)
# q  : continuous dividend yield (annual). default 0.0
#
# option_type : "call" or "put"
#
# How to call:
#
# price = black_scholes_price(
#     S0=100, K=105, T=1.0, r=0.03, sigma=0.2,
#     q=0.0, option_type="call"
# )
#
# greeks = black_scholes_greeks(
#     S0=100, K=105, T=1.0, r=0.03, sigma=0.2,
#     q=0.0, option_type="call"
# )
#
# Returns:
# price -> float
# greeks -> dict with keys: price, delta, gamma, vega, theta, rho

import numpy as np
from math import log, sqrt, exp

try:
    from scipy.stats import norm
except Exception as e:
    raise ImportError("scipy is required for Black-Scholes (scipy.stats.norm). Install scipy.") from e

def _validate_inputs(S0, K, T, r, sigma, q, option_type):
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    option_type = str(option_type).lower().strip()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    return option_type


def _d1_d2(S0, K, T, r, sigma, q):
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def black_scholes_price(S0, K, T, r, sigma, q=0.0, option_type="call"):
    option_type = _validate_inputs(S0, K, T, r, sigma, q, option_type)
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)

    if option_type == "call":
        price = S0 * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S0 * exp(-q * T) * norm.cdf(-d1)

    return float(price)


def black_scholes_greeks(S0, K, T, r, sigma, q=0.0, option_type="call"):
    option_type = _validate_inputs(S0, K, T, r, sigma, q, option_type)
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = exp(-q * T) * norm.cdf(d1)
    else:
        delta = exp(-q * T) * (norm.cdf(d1) - 1.0)

    gamma = exp(-q * T) * pdf_d1 / (S0 * sigma * sqrt(T))
    vega = S0 * exp(-q * T) * pdf_d1 * sqrt(T)

    if option_type == "call":
        theta = (
            - (S0 * exp(-q * T) * pdf_d1 * sigma) / (2.0 * sqrt(T))
            - r * K * exp(-r * T) * norm.cdf(d2)
            + q * S0 * exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * exp(-r * T) * norm.cdf(d2)
    else:
        theta = (
            - (S0 * exp(-q * T) * pdf_d1 * sigma) / (2.0 * sqrt(T))
            + r * K * exp(-r * T) * norm.cdf(-d2)
            - q * S0 * exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)

    price = black_scholes_price(S0, K, T, r, sigma, q=q, option_type=option_type)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
