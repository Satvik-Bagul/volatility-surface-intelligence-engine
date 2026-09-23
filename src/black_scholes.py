import math

def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def call_price(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = map(float, (S, K, T, r, q, sigma))
    if S <= 0 or K <= 0:
        return float("nan")
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r-q+0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*math.exp(-q*T)*_cdf(d1) - K*math.exp(-r*T)*_cdf(d2)

def put_price(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = map(float, (S, K, T, r, q, sigma))
    if S <= 0 or K <= 0:
        return float("nan")
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S/K) + (r-q+0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*_cdf(-d2) - S*math.exp(-q*T)*_cdf(-d1)
