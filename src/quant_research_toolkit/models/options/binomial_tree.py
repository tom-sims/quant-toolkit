# === BINOMIAL TREE INPUTS ===
# S0 : spot price now
# K  : strike
# T  : time to maturity in years
# r  : continuously-compounded risk-free rate (annual)
# sigma : volatility (annual)
# q  : continuous dividend yield (annual). default 0.0
#
# steps : number of binomial steps (e.g. 100-1000)
#
# option_type : "call" or "put"
# style : "european" or "american"
#
# How to call:
#
# price = binomial_tree_price(
#     S0=100, K=105, T=1.0, r=0.03, sigma=0.2,
#     q=0.0, steps=200, option_type="call", style="american"
# )
#
# Outputs:
# returns a float option price

import numpy as np

def _validate_inputs(S0, K, T, r, sigma, q, steps, option_type, style):
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    option_type = str(option_type).lower().strip()
    style = str(style).lower().strip()

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'.")

    return option_type, style


def binomial_tree_price(
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    steps=200,
    option_type="call",
    style="european",
):
    option_type, style = _validate_inputs(S0, K, T, r, sigma, q, steps, option_type, style)

    dt = T / float(steps)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if p <= 0.0 or p >= 1.0:
        raise ValueError("Risk-neutral probability out of bounds. Try more steps or check inputs.")

    j = np.arange(steps + 1)
    S_T = S0 * (u ** j) * (d ** (steps - j))

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

        if style == "american":
            j = np.arange(i + 1)
            S_i = S0 * (u ** j) * (d ** (i - j))
            if option_type == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])


def binomial_tree_greeks(
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    steps=200,
    option_type="call",
    style="european",
):
    option_type, style = _validate_inputs(S0, K, T, r, sigma, q, steps, option_type, style)

    dt = T / float(steps)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if p <= 0.0 or p >= 1.0:
        raise ValueError("Risk-neutral probability out of bounds. Try more steps or check inputs.")

    j = np.arange(steps + 1)
    S_T = S0 * (u ** j) * (d ** (steps - j))

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    value_step_1 = None
    value_step_2 = None
    S_step_1 = None
    S_step_2 = None

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

        j = np.arange(i + 1)
        S_i = S0 * (u ** j) * (d ** (i - j))

        if style == "american":
            if option_type == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)
            values = np.maximum(values, exercise)

        if i == 2:
            value_step_2 = values.copy()
            S_step_2 = S_i.copy()
        if i == 1:
            value_step_1 = values.copy()
            S_step_1 = S_i.copy()

    price = float(values[0])

    # Delta using first step up/down
    if value_step_1 is None or S_step_1 is None or len(value_step_1) < 2:
        delta = np.nan
        gamma = np.nan
    else:
        V_up = value_step_1[1]
        V_dn = value_step_1[0]
        S_up = S_step_1[1]
        S_dn = S_step_1[0]
        delta = (V_up - V_dn) / (S_up - S_dn)

        # Gamma using second step
        if value_step_2 is None or S_step_2 is None or len(value_step_2) < 3:
            gamma = np.nan
        else:
            # At i=2 we have 3 nodes: dd, ud, uu
            V_dd, V_ud, V_uu = value_step_2[0], value_step_2[1], value_step_2[2]
            S_dd, S_ud, S_uu = S_step_2[0], S_step_2[1], S_step_2[2]

            delta_up = (V_uu - V_ud) / (S_uu - S_ud)
            delta_dn = (V_ud - V_dd) / (S_ud - S_dd)

            gamma = (delta_up - delta_dn) / ((S_uu - S_dd) / 2.0)

    if value_step_1 is None:
        theta = np.nan
    else:
        v_next = disc * (p * value_step_1[1] + (1.0 - p) * value_step_1[0])
        theta = (v_next - price) / dt

    return {
        "price": price,
        "delta": float(delta) if np.isfinite(delta) else np.nan,
        "gamma": float(gamma) if np.isfinite(gamma) else np.nan,
        "theta": float(theta) if np.isfinite(theta) else np.nan,
    }
