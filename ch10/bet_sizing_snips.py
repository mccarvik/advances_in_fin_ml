import pdb
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from util.multiprocessing import mp_pandas_obj
from data.grab_data import get_google, get_tick
from ch3.labeling_snips import cusum_filter, get_t1, get_3barriers, get_bins, get_daily_vol, macd_side


def betSize(w,x): 
    return x*(w+x**2)**-.5


def getTPos(w,f,mP,maxPos): 
    return int(betSize(w,f-mP)*maxPos)
    
    
def invPrice(f,w,m): 
    return f-m*(w/(1-m**2))**.5
    
    
def limitPrice(tPos,pos,f,w,maxPos): 
    sgn=(1 if tPos>=pos else -1) 
    lP=0
    for j in xrange(abs(pos+sgn),abs(tPos+1)):
        lP += invPrice(f,w, j/ﬂoat(maxPos))
    lP/=tPos-pos 
    return lP


def getW(x,m): 
    # 0<alpha<1 
    return x**2*(m**-2-1) 


def discrete_signal(signal, step_size):
    # Discrete signal
    # eliminates all signals below step size and keeps bets in increments of the step
    pdb.set_trace()
    disc_sig = (signal / step_size).round() * step_size
    disc_sig[disc_sig > 1] = 1
    disc_sig[disc_sig < -1] = -1
    return disc_sig


def get_signal(events, step_size, prob, pred, num_classes, num_threads, **kwargs):
    # Get signals from predictions
    # prob = probility of label x taking places
    # pred = the predicted value --> {-1, 1}
    if prob.shape[0] == 0:
        return pd.Series()
    # Generate signals from multinomial
    # Get the z score
    signal0 = (prob - 1. / num_classes) / np.sqrt(prob * (1. - prob))
    # Derive the bet size based on strength of signal (pred dictates side i.e. long / short)
    signal0 = pred * (2 * norm.cdf(signal0) - 1)
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']
    # Averaging
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avg_active_signals(df0, num_threads)
    signal1 = discrete_signal(signal=df0, step_size=step_size)
    return signal1


def mp_avg_active_signals(signals, molecule):
    out = pd.Series()
    for loc in molecule:
        is_act = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[is_act].index
        if len(act) > 0:
            # averages the signals across all active signals
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0
    return out


def avg_active_signals(signals, num_threads):
    # Compute the average signal
    # 1) time points where singal changes
    t_pnts = set(signals['t1'].dropna().values)
    t_pnts = t_pnts.union(signals.index.values)
    t_pnts = list(t_pnts)
    t_pnts.sort();
    out = mp_pandas_obj(mp_avg_active_signals, ('molecule', t_pnts), num_threads, signals=signals)
    return out
    

def demo():
    samples = np.random.uniform(.5, 1., 10000)
    zs = (samples - .5) / np.sqrt(samples * (1 - samples))
    bet_size = 2 * norm.cdf(zs) - 1
    print(bet_size)


def demo2(): 
    # adjust bet sizes as market price and forecast price fluctuate
    pos,maxPos,mP,f,wParams=0,100,100,115,{'divergence':10,'m':.95} 
    w=getW(wParams['divergence'],wParams['m']) # calibrate w 
    tPos=getTPos(w,f,mP,maxPos) # get tPos 
    LP=limitPrice(tPos,pos,f,w,maxPos) # limit price for order 
    return


def demo3():
    close = get_tick('AAL')
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=7)
    side =  macd_side(close)
    events = get_3barriers(close, t_events=sampled_idx, trgt=vol, ptsl=[1, 2], t1=t1, side=side)
    events = events.dropna()
    bins = get_bins(events, close)
    
    clf = RandomForestClassifier()
    x = np.hstack([events['side'].values[:, np.newaxis], close.loc[events.index].values[:, np.newaxis]])  # action and px
    # if return was positive, bins = 1
    y = bins['bin'].values  # supervised answer
    clf.fit(x, y)
    predicted_probs = np.array([x[1] for x in clf.predict_proba(x)])

    # get_signal(events.drop(columns=['side']), 0.2, predicted_probs, events['side'], 2, 1)
    get_signal(events.drop(columns=['side']), 0.2, predicted_probs, events['side'], 2, 12)


if __name__ == '__main__':
    # demo()
    # demo2()
    demo3()