import pdb
import sys
import pandas as pd
import numpy as np
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from ch10.bet_sizing_snips import getTPos, getW
from data.grab_data import get_tick


def getHoldingPeriod(tPos):
	# Derive avg holding period (in days) using avg entry time pairing algo
	hp, tEntry = pd.DataFrame(columns=['dT', 'w']), 0.
	# pDiff, tDiff = tPos.diff(), (tPos.index - tPos.index[0]) / np.timedelta64(1, 'D')
	# just using ints instead of date
	pDiff, tDiff = tPos.diff(), (tPos.index - tPos.index[0]) / 1
	for i in range(1, tPos.shape[0]):
		# Target position has increased or stayed the same
		# absolute value --> short position got MORE short or stayed the same
		if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0:
			if tPos.iloc[i] != 0:
				# reset entry time of position weighted by position
				tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
		else:
			# Target position has decreased --> new holding period
			if tPos.iloc[i] * tPos.iloc[i - 1] < 0:
				# target position has flipped
				hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
				# reset entry time of position
				tEntry = tDiff[i]
			else:
				# target position has become smaller
				hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))
	if hp['w'].sum() > 0:
		# get average weighted holding period
		hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum()
	else:
		hp = np.nan
	return hp


def getHHI(betRet):
	if betRet.shape[0] <= 2:
		return np.nan
	wght = betRet / betRet.sum()
	# more spread out returns will be minimized by the squared weight
	hhi = (wght ** 2).sum()
	hhi = (hhi - betRet.shape[0] ** -1) / (1. - betRet.shape[0] ** -1)
	return hhi


def computeDD_TuW(series, dollars=False):
	# SNIPPET 14.4 DERIVING THE SEQUENCE OF DD AND TuW
	# compute series of drawdowns and the time under water associated with them
	# drawdown (DD) --> max loss suffered by an investment between two consecutive high-watermarks (HWMs)
	# Time under water (TUW) --> time elapsed between an HWM and the moment the PnL exceeds the previous maximum PnL
	# df0 = series.to_frame('pnl')
	df0 = pd.DataFrame(data={'pnl': series - series.iloc[0]})
	df0['hwm'] = series.expanding().max()
	# min value during this high water mark
	df1 = df0.groupby('hwm').min().reset_index()
	df1.columns = ['hwm', 'min']
	df1.index = df0['hwm'].drop_duplicates(keep='first').index  # time of hwm
	df1 = df1[df1['hwm'] > df1['min']]  # hwm followed by a drawdown
	if dollars:
		dd = df1['hwm'] - df1['min']
	else:
		dd = 1 - df1['min'] / df1['hwm']
	tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'Y')).values  # in years
	tuw = pd.Series(tuw, index=df1.index[:-1])
	return dd, tuw


def demo():
	# SNIPPET 14.1 DERIVING THE TIMING OF BETS FROM A SERIES OF TARGET POSITIONS
    tPos = pd.Series([4, 3, -2, 1, 0, 6, 8, 0, -5, 7])
    
    # A bet takes place between flat positions or position flips
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    df1 = df1[df1 != 0].index
    # all the times the position went from 0 to a non zero position
    bets = df0.intersection(df1)  # flattening
    df0 = tPos.iloc[1:] * tPos.iloc[:-1].values
    # all the times the position flip from short to long or vice versa
    bets = bets.union(df0[df0 < 0].index).sort_values()  # tPos flips
    if tPos.index[-1] not in bets:
        bets = bets.append(tPos.index[-1:])  # last bet
    print(bets)


def demo2():
	# SNIPPET 14.2 IMPLEMENTATION OF A HOLDING PERIOD ESTIMATOR
	# tPos = pd.Series([4, 3, -2, 1, 0, 6, 8, 0, -5, 7])
	# getHoldingPeriod(tPos)
	
	# SNIPPET 14.3 ALGORITHM FOR DERIVING HHI CONCENTRATION
	close = get_tick('AAL')
	ret = close.diff().values
	ret = ret[~np.isnan(ret)]
	pdb.set_trace()
	rHHIPos = getHHI(ret[ret >= 0])  # concentration of positive returns per bet
	print(rHHIPos)
	rHHINeg = getHHI(ret[ret < 0])  # concentration of negative returns per bet
	print(rHHINeg)
	tHHI = getHHI(close.groupby(pd.Grouper(freq='M')).count())  # concentr. bets/month
	print(tHHI)
	
	
def demo3():
	close = get_tick('FB')
	computeDD_TuW(close)
	
	
if __name__ == '__main__':
	# demo()
	# demo2()
	demo3()