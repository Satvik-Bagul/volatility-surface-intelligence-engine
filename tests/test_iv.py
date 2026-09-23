import numpy as np
from src.black_scholes import call_price
from src.iv import implied_volatility

def test_iv_round_trip():
    p=call_price(100,100,1,.05,0,.20)
    iv=implied_volatility(p,100,100,1,.05,0,"call")
    assert abs(iv-.20)<1e-5

def test_invalid_price():
    assert np.isnan(implied_volatility(-1,100,100,1,.05,0,"call"))
