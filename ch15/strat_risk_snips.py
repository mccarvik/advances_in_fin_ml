import pdb
import sys
import pandas as pd
import numpy as np
import scipy.stats as ss
from sympy import init_printing, symbols, factor
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_tick


def binHR(sl, pt, freq, tSR):
    # SNIPPET 15.3 COMPUTING THE IMPLIED PRECISION
	"""
	Given a trading rule characterized by the parameters {sl,pt,freq},
	what's the min precision p required to achieve a Sharpe ratio tSR?
	1) Inputs
	sl: stop loss threshold
	pt: profit taking threshold
	freq: number of bets per year
	tSR: target annual Sharpe ratio
	2) Output
	p: the min precision rate p required to achieve tSR
	"""
	a = (freq + tSR ** 2) * (pt - sl) ** 2
	b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
	c = freq * sl ** 2
	p = (-b + (b ** 2 - 4 * a * c) ** .5) / (2. * a)
	return p


def binFreq(sl, pt, p, tSR):
    # SNIPPET 15.4 COMPUTING THE IMPLIED BETTING FREQUENCY
	"""
	Given a trading rule characterized by the parameters {sl,pt,freq},
	what's the number of bets/year needed to achieve a Sharpe ratio
	tSR with precision rate p?
	Note: Equation with radicals, check for extraneous solution.
	1) Inputs
	sl: stop loss threshold
	pt: profit taking threshold
	p: precision rate p
	tSR: target annual Sharpe ratio
	2) Output
	freq: number of bets per year needed
	"""
	freq = (tSR * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2  # possible extraneous
    # if not np.isclose(binHR(sl, pt, freq, p), tSR):
    # 	return  # TODO: MAYBE IT SHOULD BE binHR instead of binSR
	return freq


def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs):
	# Random draws from a mixture of gaussians
    # mus / sigmas --> mean and stdev of gaussian random variables
    # nobs --> number of samples, prob --> probability of a win of a given sample
    
    # average wins and number of wins based on prob
	ret1 = np.random.normal(mu1, sigma1, size=int(nObs * prob1))
	# average losses and number of losses based on prob
	ret2 = np.random.normal(mu2, sigma2, size=int(nObs) - ret1.shape[0])
	ret = np.append(ret1, ret2, axis=0)
	np.random.shuffle(ret)
	return ret


def probFailure(ret, freq, tSR):
	# Derive probability that strategy may fail
    # average loss and average win
	rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
	p = ret[ret > 0].shape[0] / float(ret.shape[0])
    # pass in average returns of wins and loss as profit take and stop loss and see if covers the target sharp
	thresP = binHR(rNeg, rPos, freq, tSR)
	risk = ss.norm.cdf(thresP, p, p * (1 - p))  # approximation to bootstrap
	return risk


def demo():
    # SNIPPET 15.1 TARGETING A SHARPE RATIO AS A FUNCTION OF THE NUMBER OF BETS
    # p (precision) = odds of correct bet
    # less frequebt bets need higher precision to acheive a desired sharpe ratio
    out, p = [], .55
    for i in range(1000000):
        rnd = np.random.binomial(n=1, p=p)
        x = (1 if rnd == 1 else -1)
        out.append(x)
    # printing out the mean, stdev and sharp ratio 	
    print(np.mean(out), np.std(out), np.mean(out) / np.std(out))
    
    
def demo2():
    # SNIPPET 15.2 USING THE SymPy LIBRARY FOR SYMBOLIC OPERATIONS
    init_printing(use_unicode=False, wrap_line=False, no_global=True)
    # Given a trading rule characterized by parameters (p u d)
    # what is the precision rate p required to achieve a Sharpe ratio of x
    p, u, d = symbols('p u d')
    m2 = p * u ** 2 + (1 - p) * d ** 2
    m1 = p * u + (1 - p) * d
    v = m2 - m1 ** 2
    print(factor(v))


def demo3():
    # necessary precision given number of bets
    print(binHR(-0.1, 0.1, 50, 1.0))
    # necessary number of bets given precision
    print(binFreq(-0.1, 0.1, 0.56, 1.0))


def demo4():
    # SNIPPET 15.5 CALCULATING THE STRATEGY RISK IN PRACTICE
    # Strategy risk is the risk that the investment strategy will fail to succeed over time
    # 1) Parameters
    mu1, mu2, sigma1, sigma2, prob1, nObs = .05, -.1, .05, .1, .75, 2600
    tSR, freq = 2., 260
    # 2) Generate sample from mixture
    ret = mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs)
    # 3) Compute prob failure
    probF = probFailure(ret, freq, tSR)
    print('Prob strategy will fail', probF)


if __name__ == '__main__':
    # demo()
    # demo2()
    # demo3()
    demo4()