import pdb
import sys
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement, product
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_tick, get_google_all


def pigeonHole(k, n):
    # SNIPPET 21.1 PARTITIONS OF k OBJECTS INTO n SLOTS
	# Pigeonhole problem (organize k objects in n slots)
	for j in combinations_with_replacement(range(n), k):
		r = [0] * n
		for i in j:
			r[i] += 1
		yield r


def getAllWeights(k, n):
    # SNIPPET 21.2 SET 𝛀 OF ALL VECTORS ASSOCIATED WITH ALL PARTITIONS
	# 1) Generate partitions
	parts, w = pigeonHole(k, n), None
	# 2) Go through partitions
	for part_ in parts:
		w_ = np.array(part_) / float(k)  # abs(weight) vector
		for prod_ in product([-1, 1], repeat=n):  # add sign
			w_signed_ = (w_ * prod_).reshape(-1, 1)
			if w is None:
				w = w_signed_.copy()
			else:
				w = np.append(w, w_signed_, axis=1)
	return w


def evalTCosts(w, params):
    # SNIPPET 21.3 EVALUATING ALL TRAJECTORIES
	# Compute t-costs of a particular trajectory
	tcost = np.zeros(w.shape[1])
	w_ = np.zeros(shape=w.shape[0])
	for i in range(tcost.shape[0]):
		c_ = params[i]['c']
		tcost[i] = (c_ * abs(w[:, i] - w_) ** .5).sum()
		w_ = w[:, i].copy()
	return tcost


def evalSR(params, w, tcost):
	# Evaluate Sharpe Ratio over multiple horizons
	mean, cov = 0, 0
	for h in range(w.shape[1]):
		params_ = params[h]
		mean += np.dot(w[:, h].T, params_['mean'])[0] - tcost[h]
		cov += np.dot(w[:, h].T, np.dot(params_['cov'], w[:, h]))
	sr = mean / cov ** .5
	return sr


def dynOptPort(params, k=None):
	# Dynamic optimal portfolio
	# 1) Generate partitions
	if k is None:
		k = params[0]['mean'].shape[0]
	n = params[0]['mean'].shape[0]
	w_all, sr = getAllWeights(k, n), None
	# 2) Generate trajectories as cartesian products
	for prod_ in product(w_all.T, repeat=len(params)):
		w_ = np.array(prod_).T  # concatenate product into a trajectory
		tcost_ = evalTCosts(w_, params)
		sr_ = evalSR(params, w_, tcost_)  # evaluate trajectory
		if sr is None or sr < sr_:  # store trajectory if better
			sr, w = sr_, w_.copy()
	return w


def rndMatWithRank(nSamples, nCols, rank, sigma=0, homNoise=True):
	# Produce random matrix X with given rank
	# SNIPPET 21.4 PRODUCE A RANDOM MATRIX OF A GIVEN RANK
	rng = np.random.RandomState()
	U, _, _ = np.linalg.svd(rng.randn(nCols, nCols))
	x = np.dot(rng.randn(nSamples, rank), U[:, :rank].T)
	if homNoise:
		x += sigma * rng.randn(nSamples, nCols)  # Adding homoscedastic noise
	else:
		sigmas = sigma * (rng.rand(nCols) + .5)  # Adding heteroscedastic noise
		x += rng.randn(nSamples, nCols) * sigmas
	return x


def genMean(size):
    # SNIPPET 21.5 GENERATE THE PROBLEM’S PARAMETERS
	# Generate a random vector of means
	rMean = np.random.normal(size=(size, 1))
	return rMean


def statOptPortf(cov, a):
	# Static optimal porftolio
	# SNIPPET 21.6 COMPUTE AND EVALUATE THE STATIC SOLUTION
	# Solution to the "unconstrained" portfolio optimization problem
	cov_inv = np.linalg.inv(cov)
	w = np.dot(cov_inv, a)
	w /= np.dot(np.dot(a.T, cov_inv), a)  # np.dot(w.T,a)==1
	w /= abs(w).sum()  # re-scale for full investment
	return w
	

def demo():
	# 1) Parameters
	size, horizon = 3, 2
	params = []
	for h in range(horizon):
		x = rndMatWithRank(1000, 3, 3, 0)
		mean_, cov_ = genMean(size), np.cov(x, rowvar=False)
		c_ = np.random.uniform(size=cov_.shape[0]) * np.diag(cov_) ** .5
		params.append({'mean': mean_, 'cov': cov_, 'c': c_})
    # 	print(params)
	return params


def demo2(params):
	# 2) Static optimal portfolios
	w_stat = None
	for params_ in params:
		w_ = statOptPortf(cov=params_['cov'], a=params_['mean'])
		if w_stat is None:
			w_stat = w_.copy()
		else:
			w_stat = np.append(w_stat, w_, axis=1)
	tcost_stat = evalTCosts(w_stat, params)
	sr_stat = evalSR(params, w_stat, tcost_stat)
	print('static SR:', sr_stat)


def demo3(params):
    # SNIPPET 21.7 COMPUTE AND EVALUATE THE DYNAMIC SOLUTION
	# 3) Dynamic optimal portfolios
	w_dyn = dynOptPort(params)
	tcost_dyn = evalTCosts(w_dyn, params)
	sr_dyn = evalSR(params, w_dyn, tcost_dyn)
	print('dynamic SR:', sr_dyn)


if __name__ == '__main__':
    params = demo()
    demo2(params)
    demo3(params)