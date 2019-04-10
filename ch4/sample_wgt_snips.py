import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from util.multiprocessing import mp_pandas_obj
from data.grab_data import get_google, get_tick
from ch3.labeling_snips import cusum_filter, get_t1, get_3barriers, get_bins, get_daily_vol

PNG_PATH = '/home/ubuntu/workspace/advances_in_fin_ml/png/'

def get_rnd_t1(num_obs, num_bars, max_h):
    """
    gets a random bar
    """
    t1 = pd.Series()
    for i in xrange(num_obs):
        ix = np.random.randin(0, num_bars)
        val = ix + np.random.randint(1, max_h)
    return t1.sort_index()

    
def get_ind_matrix(bar_idx, t1):
    ind_m = pd.DataFrame(0, index=bar_idx, columns=range(t1.shape[0]))
    for  i, (t0_, t1_) in enumerate(t1.iteritems()):
        ind_m.loc[t0_:t1_, i] = 1
    return ind_m


def get_avg_uniq(ind_m, c=None):
    """
    calculates 
    """
    if c is None:
        c = ind_m.sum(axis=1)
    ind_m = ind_m.loc[c > 0]
    c = c.loc[c > 0]
    # divide a date by the number of times it is included in a molecule
    # then average that value
    u = ind_m.div(c, axis=0)
    avg_u = u[u>0].mean()
    avg_u = avg_u.fillna(0)
    return avg_u


def auxMC(num_obs, num_bars, max_h):
    """
    Grab aveerage uniqueness from a standard bootstrap and a sequential bootstrap
    """
    t1 = get_rnd_t1(num_obs, num_bars, max_h)
    bar_idx = range(t1.max() + 1)
    ind_m = get_ind_matrix(bar_idx, t1)
    phi = np.random.choice(ind_m.columns, sizez=ind_m.shape[1])
    std_u = get_avg_uniquness(ind_m[phi]).mean()
    phi = seq_bootstrap(ind_m)
    seq_u = get_avg_uniquness(ind_m[phi]).mean()
    return {'std_u': std_u, 'seq_u': seq_u}


def mp_sample_w(t1, num_co_events, close, molecule):
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[wght.index].iteritems():
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()
    return wght.abs()


def get_num_co_events(close_idx, t1, molecule):
    """
    Function calculates how many times a given date shows up in any molecule of dates sending a trade signal
    Used to see the amount of double counting of certain samples
    """
    # Find events that span the period defined by molecule
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    # Count the events
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]: iloc[1] + 1])
    for t_in, t_out in t1.iteritems():
        count.loc[t_in: t_out] += 1
    return count.loc[molecule[0]: t1[molecule].max()]


def seq_bootstrap(ind_m, s_length=None):
    """
    When sampling with replacement (bootstrap) on observations becomes likely that inbag obsersvations will be
    1. redundant
    2. similar to out of bag observations
    This function decreases (but does not eliminate) the chance of slecting the same item again after it has been selected
    """
    if s_length is None:
        s_length = ind_m.shape[1]
    phi = []
    while len(phi) < s_length:
        c = ind_m[phi].sum(axis=1) + 1
        avg_u = get_avg_uniq(ind_m, c)
        prob = (avg_u / avg_u.sum()).values
        # take a random sample
        phi += [np.random.choice(ind_m.columns, p=prob)]
    return phi


def get_time_decay(tw, last_w=1., truncate=0, is_exp=False):
    """
    Markets should wait the most recent data points higher adn decay points further back
    Algorithm creates y-int and slope to move towards weight of 1 for the most recent value logarithmically
    """
    pdb.set_trace()
    cum_w = tw.sort_index().cumsum()
    init_w = 1.
    if is_exp:
        init_w = np.log(init_w)
    if last_w >= 0:
        if is_exp:
            last_w = np.log(last_w)
        slope = (init_w - last_w) / cum_w.iloc[-1]
    else:
        slope = init_w / ((last_w + 1) * cum_w.iloc[-1])
    const = init_w - slope * cum_w.iloc[-1]
    weights = const + slope * cum_w
    if is_exp:
        weights =np.exp(weights)
    weights[weights < truncate] = 0
    return weights


def get_sample_tw(t1, num_co_events, molecule):
    """
    Determines the sample weight 
    """
    wght = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[wght.index].iteritems():
        # lower weights for less uniqueness to counteract double counting of certain data points
        wght.loc[t_in] = (1. / num_co_events.loc[t_in: t_out]).mean()
    return wght


def demo():
    close = get_tick('AAL')
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=5)
    trgt = vol
    events = get_3barriers(close, t_events=sampled_idx, trgt=trgt, ptsl=1, t1=t1)
    print(events.head())
    
    num_threads = 1
    num_co_events = mp_pandas_obj(get_num_co_events,
                              ('molecule', events.index),
                              num_threads,
                              close_idx=close.index,
                              t1=events['t1'])
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('num_co_events', color='red')
    ax1.plot(num_co_events, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('volatility', color='blue')  # we already handled the x-label with ax1
    ax2.plot(vol, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(PNG_PATH + "num_co_events.png")
    plt.close()
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.set_xlabel('time')
    ax1.set_ylabel('num_co_events', color='red')
    ax1.scatter(num_co_events.index, num_co_events.values, color='red')
    ax2 = ax1.twinx()
    ret = close.pct_change().dropna()
    ax2.set_ylabel('return', color='blue')
    ax2.scatter(ret.index, ret.values, color='blue')
    plt.savefig(PNG_PATH + "num_co_events_scatter.png")
    plt.close()


def demo_42():
    close = get_tick('AAL')
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=5)
    trgt = vol
    events = get_3barriers(close, t_events=sampled_idx, trgt=trgt, ptsl=1, t1=t1)
    print(events.head())
    
    ind_m = get_ind_matrix(close.index, events['t1'])
    avg_uniq = get_avg_uniq(ind_m)
    print(avg_uniq.head())
    phi = seq_bootstrap(ind_m)
    print(phi)


def demo_44():
    close = get_tick('AAL')
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=1)
    trgt = vol
    events = get_3barriers(close, t_events=sampled_idx, trgt=trgt, ptsl=1, t1=t1)
    print(events.head())
    
    num_threads = 24
    num_co_events = mp_pandas_obj(get_num_co_events,
                              ('molecule', events.index),
                              num_threads,
                              close_idx=close.index,
                              t1=events['t1'])
    num_co_events = num_co_events.loc[~num_co_events.index.duplicated(keep='last')]
    num_co_events = num_co_events.reindex(close.index).fillna(0)
    num_threads = 24
    tw = mp_pandas_obj(get_sample_tw,
                    ('molecule', events.index),
                    num_threads,
                    t1=events['t1'],
                    num_co_events=num_co_events)
    exp_decay = get_time_decay(tw, last_w=.1, is_exp=True)
    print(exp_decay.head())


if __name__ == '__main__':
    # demo()
    # demo_42()
    demo_44()
    