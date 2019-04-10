import pdb
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_tick, get_google_all


# SNIPPET 19.1 IMPLEMENTATION OF THE CORWIN-SCHULTZ ALGORITHM
def getBeta(series, sl):
    # calcualte beta variable in calculation
    hl = series[['High', 'Low']].values
    hl = np.log(hl[:, 0] / hl[:, 1]) ** 2
    hl = pd.Series(hl, index=series.index)
    beta = hl.rolling(2).sum()
    beta = beta.rolling(sl).mean()
    return beta.dropna()


# ---
def getGamma(series):
    # calcualte gamma variable in calculation
    h2 = series['High'].rolling(2).max()
    l2 = series['Low'].rolling(2).max()
    gamma = np.log(h2.values / l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()


# ---
def getAlpha(beta, gamma):
    # calcualte gamma variable in calculation
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # set negative alphas to 0 (see p.727 of paper)
    return alpha.dropna()


def getSigma(beta, gamma):
    # sigma variable representing volatility needed for Becker-Parkinson volatility approach
    pdb.set_trace()
    k2 = (8 / np.pi) ** .5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -.5 - 1) * beta ** .5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** .5
    sigma[sigma < 0] = 0
    return sigma


# ---
def corwinSchultz(series, sl=1):
    # Note: S<0 iif alpha<0
    beta = getBeta(series, sl)
    gamma = getGamma(series)
    alpha = getAlpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    startTime = pd.Series(series.index[0:spread.shape[0]], index=spread.index)
    pdb.set_trace()
    spread = pd.concat([spread, startTime], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    
    # Becker Parkinson Volatility
    sigma = getSigma(beta, gamma)
    return spread


def demo():
    close = get_google_all()
    corwinSchultz(close)



if __name__ == '__main__':
    demo()