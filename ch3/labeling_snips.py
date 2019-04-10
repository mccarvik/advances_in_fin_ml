import pdb
import sys
import numbers
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_google, get_tick


def get_daily_vol(close, span=100):
    use_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    use_idx = use_idx[use_idx > 0]
    # Get rid of duplications in index
    use_idx = np.unique(use_idx)
    prev_idx = pd.Series(close.index[use_idx - 1], index=close.index[use_idx])
    ret = close.loc[prev_idx.index] / close.loc[prev_idx.values].values - 1
    # gets the standard deviation on a one day return using the mean from the past span (default=100) days
    vol = ret.ewm(span=span).std()
    return vol


def cusum_filter(close, h):
    # asssum that E y_t = y_{t-1}
    t_events = []
    s_pos, s_neg = 0, 0
    ret = close.pct_change().dropna()
    # gets the diff of returns from one day to the next
    diff = ret.diff().dropna()
    # time variant threshold
    if isinstance(h, numbers.Number):
        h = pd.Series(h, index=diff.index)
    h = h.reindex(diff.index, method='bfill')
    h = h.dropna()
    # Finds moments when the cumulative return change is above or below the threshold h in return
    # then set reference point back to 0
    for t in h.index:
        s_pos = max(0, s_pos + diff.loc[t])
        s_neg = min(0, s_neg + diff.loc[t])
        if s_pos > h.loc[t]:
            s_pos = 0
            t_events.append(t)
        elif s_neg < -h.loc[t]:
            s_neg = 0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)


def get_t1(close, t_events, num_days):
    # sets the vertical barrier
    # For each index in tEvents finds the timestamp of the next price bar at or immediately after a number of numDays
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    return t1


def get_3barriers(close, t_events, ptsl, trgt, min_ret=0, num_threads=1, t1=False, side=None):
    # PTSL = profit taking / stop losses
    # Get sampled target values
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    # Get time boundary t1
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    # Define the side
    if side is None:
        _side = pd.Series(1., index=trgt.index)
        _ptsl = [ptsl, ptsl]
    else:
        # if only one horizontal barrier
        _side = side.loc[trgt.index]
        _ptsl = ptsl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
    events = events.dropna(subset=['trgt'])
    time_idx = apply_ptslt1(close, events, _ptsl, events.index)
    # Skip when all of barrier are not touched
    # take whatever PT or SL that happened first
    time_idx = time_idx.dropna(how='all')
    events['t1_type'] = time_idx.idxmin(axis=1)
    events['t1'] = time_idx.min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events

    
def apply_ptslt1(close, events, ptsl, molecule):
    """Return dataframe about if price touches the boundary"""
    # Sample a subset with specific indices
    _events = events.loc[molecule]
    # Time limit
    
    out = pd.DataFrame(index=_events.index)
    # Set Profit Taking and Stop Loss
    if ptsl[0] > 0:
        # Profit taking on - get horizontal top barrier
        pt = ptsl[0] *  _events["trgt"]
    else:
        # Switch off profit taking
        pt = pd.Series(index=_events.index)
    if ptsl[1] > 0:
        # stop loss on - get horizontal bottom barrier
        sl = -ptsl[1] * _events["trgt"]
    else:
        # Switch off stop loss
        sl = pd.Series(index=_events.index)
        
    # Replace undifined value with the last time index
    # loc and t1 are the start and end of the period
    time_limits = _events["t1"].fillna(close.index[-1])
    for loc, t1 in time_limits.iteritems():
        df = close[loc:t1]
        # Change the direction depending on the side - df gets the return
        df = (df / close[loc] - 1) * _events.at[loc, 'side']
        print(loc, t1, df[df < sl[loc]].index.min(), df[df > pt[loc]].index.min())
        # check if profit taking or stop loss is hit
        out.at[loc, 'sl'] = df[df < sl[loc]].index.min()
        out.at[loc, 'pt'] = df[df > pt[loc]].index.min()
    out['t1'] = _events['t1'].copy(deep=True)
    # returns 4 columns - start, stop loss if any, profit take if any, end
    return out    


def get_bins(events, close):
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    px = events.index.union(events['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index] - 1.
    if 'side' in events:
        out['ret'] *= events['side']
    out['bin'] = np.sign(out['ret'])
    # 0 when touching vertical line, type = t1
    out['bin'].loc[events['t1_type'] == 't1'] = 0
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    return out
    

def drop_labels(events, min_pct=0.05):
    while True:
        # drops any labels below certain threshold 
        df = events['bin'].value_counts(normalize=True)
        if df.min() > min_pct or df.shape[0] < 3:
            break
        print('dropped label', df.argmin(), df.min())
        events = events[events['bin'] != df.argmin()]
    return events


def macd_side(close):
    # Moving Average Convergence Divergence (MACD)
    # https://www.investopedia.com/terms/m/macd.asp
    # macd = 12  history = 26 days  signal = 9 days
    macd, signal, hist = talib.MACD(close.values)
    hist = pd.Series(hist).fillna(1).values
    return pd.Series(2 * ((hist > 0).astype(float) - 0.5), index=close.index[-len(hist):])


def demo():
    close = get_tick('AAL')
    
    # Daily Volatility
    vol = get_daily_vol(close)
    # print(vol.head())
    
    # cusum filter
    # cusum = cusum_filter(close, 0.1)
    sampled_idx  = cusum_filter(close, vol)
    # print(sampled_idx)
    
    # get vertical barrier
    t1 = get_t1(close, sampled_idx, num_days=7)
    # print(t1.head())
    
    # gets the events and which barrier was hit
    # ptsl = 1 for long ptsl = -1 for short
    events = get_3barriers(close, t_events=sampled_idx, trgt=vol, ptsl=1, t1=t1)
    print(events.head())
    # print(events['t1_type'].unique())
    print(events['t1_type'].describe())

    
    # returns 2 columsn, bin (profit or loss or timout) and return
    bins = get_bins(events, close)
    # print(bins)
    # print(bins['bin'].value_counts())
    
    dropped_bins = drop_labels(bins)
    # print(bins.shape)
    print(dropped_bins.head())


def macd_demo():
    close = get_tick('AAL')
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=7)
    side =  macd_side(close)
    events = get_3barriers(close, t_events=sampled_idx, trgt=vol, ptsl=[1, 2], t1=t1, side=side)
    events = events.dropna()
    # print(events.head())
    bins = get_bins(events, close)
    # print(bins.head())
    
    
    clf = RandomForestClassifier()
    x = np.hstack([events['side'].values[:, np.newaxis], close.loc[events.index].values[:, np.newaxis]])  # action and px
    y = bins['bin'].values  # supervised answer
    clf.fit(x, y)
    pred = clf.predict(x)
    # As dictated by MACD indicator
    # print(events['side'].values)
    # print(help(talib.MACD))
    macd, signal, hist = talib.MACD(close.values)
    print(np.max(macd[100:] - signal[100:]  - hist[100:] ))
    print(macd[np.isfinite(macd)].shape)
    signal = signal[np.isfinite(signal)]
    print(2 * ((signal > 0).astype(float) - 0.5))
    macd.fill(1)
    print(macd)
    

if __name__ == '__main__':
    # demo()
    macd_demo()
    
    
    
    
    