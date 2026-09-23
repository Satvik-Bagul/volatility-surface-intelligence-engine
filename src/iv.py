import numpy as np
from scipy.optimize import brentq
from .black_scholes import call_price, put_price

def bounds(S,K,T,r,q,typ):
    ds=S*np.exp(-q*T); dk=K*np.exp(-r*T)
    return ((max(0,ds-dk),ds) if typ=="call" else (max(0,dk-ds),dk))

def implied_volatility(price,S,K,T,r,q,typ,high=2.0,max_high=8.0):
    vals=[price,S,K,T,r,q]
    if not np.all(np.isfinite(vals)) or price<=0 or S<=0 or K<=0 or T<=0:
        return np.nan
    lo_bound,hi_bound=bounds(S,K,T,r,q,typ)
    if price <= lo_bound+1e-8 or price >= hi_bound-1e-8:
        return np.nan
    fn=call_price if typ=="call" else put_price
    def f(sig): return fn(S,K,T,r,q,sig)-price
    low=1e-5
    try:
        flo=f(low); fhi=f(high)
        while flo*fhi>0 and high<max_high:
            high*=2; fhi=f(high)
        if flo*fhi>0: return np.nan
        return float(brentq(f,low,high,xtol=1e-8,maxiter=100))
    except (ValueError,RuntimeError,OverflowError):
        return np.nan

def add_iv(df, rate=0.045, dividend_yield=0.0):
    df=df.copy()
    df["r"]=float(rate); df["q"]=float(dividend_yield)
    df["calculated_iv"]=[implied_volatility(x.mid,x.spot,x.strike,x.T,x.r,x.q,x.option_type) for x in df.itertuples()]
    return df
