import pdb
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_tick
from util.utils import PNG_PATH, DATA_PATH


# SNIPPET 17.1 SADF’S INNER LOOP
def get_bsadf(logP, minSL, constant, lags):
	# SADF = Supremum Augmented Dickey-Fuller 
	y, x = getYX(logP, constant=constant, lags=lags)
	startPoints, bsadf, allADF = range(0, y.shape[0] + lags - minSL + 1), None, []
	for start in startPoints:
		y_, x_ = y[start:], x[start:]
		bMean, bStd = getBetas(y_, x_)
		# get most recent beta nd stdev
		bMean, bStd = bMean[0, 0], bStd[0, 0] ** .5
		allADF.append(bMean / bStd)
		# set bsadf to conduct SADF test
		if not bsadf or allADF[-1] > bsadf:
			bsadf = allADF[-1]
	out = {'Time': logP.index[-1], 'gsadf': bsadf}
	return out


# SNIPPET 17.2 PREPARING THE DATASETS
def getYX(series, constant, lags):
	series_ = series.diff().dropna()
	x = lagDF(series_, lags).dropna()
	x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0]  # lagged level
	y = series_.iloc[-x.shape[0]:].values
	# adds an index numnber and a column of 1s needed for the regeression tests
	if constant != 'nc':
		x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
		trend = 0
		if constant[:2] == 'ct':
			trend = np.arange(x.shape[0]).reshape(-1, 1)
			x = np.append(x, trend, axis=1)
		if constant == 'ctt':
			x = np.append(x, trend ** 2, axis=1)
	return y, x


# SNIPPET 17.3 APPLY LAGS TO DATAFRAME
def lagDF(df0, lags):
	# creates a lag dataframe with prices lagged by 'lags' number of days
	df1 = pd.DataFrame()
	if isinstance(lags, int):
		lags = range(lags + 1)
	else:
		lags = [int(lag) for lag in lags]
	for lag in lags:
		df_ = df0.shift(lag).copy(deep=True)
		df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
		df1 = df1.join(df_, how='outer')
	return df1


# SNIPPET 17.4 FITTING THE ADF SPECIFICATION
def getBetas(y, x):
	# calculate the mean beta and the variance
	xy = np.dot(x.T, y)
	xx = np.dot(x.T, x)
	xxinv = np.linalg.inv(xx)
	bMean = np.dot(xxinv, xy)
	err = y - np.dot(x, bMean)
	bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
	return bMean, bVar
	

def demo():
	close = get_tick('AAL').to_frame()
	close['pct_change'] = close['px'].pct_change()
	close['log_ret'] = np.log(close['px']) - np.log(close['px'].shift(1))
	print(get_bsadf(close['log_ret'].to_frame(), 50, 'ct', 10))

	
if __name__ == '__main__':
    demo()
    