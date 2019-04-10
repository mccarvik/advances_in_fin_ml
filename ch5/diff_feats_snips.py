import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from util.utils import PNG_PATH
from data.grab_data import get_google, get_tick
from ch3.labeling_snips import cusum_filter, get_t1, get_3barriers, get_daily_vol


# Nonstaionary data - mean (and maybe variance0 change over the course of the data
#   ex: stock price goes up over time



def get_weights(d, size):
    """
    Get weights so that the mean is 0
    Original weight will be one and then every weight behind it will be negative and fractional
    Will cause stationarity
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def get_weights_FFD(d, thres, size=10000):
    """
    Will break when the weight gets below a certain threshold
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) <= thres:
            break
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plot_weights(d_range, n_plots, size):
    w = pd.DataFrame()
    for d in np.linspace(d_range[0], d_range[1], n_plots):
        w_ = get_weights(d, size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot(figsize=(16, 8))
    ax.legend(loc='upper right')
    plt.savefig(PNG_PATH + "plot_weights_{}{}.png".format(d_range[0], d_range[1]))
    plt.close()


def frac_diff(series, d, thres=.1):
    # w.shape = (series.shape[0], 1)
    w = get_weights(d, series.shape[0])
    w_sum = np.cumsum(abs(w))
    w_sum /= w_sum[-1]
    # Usable only after going over the threshold
    skip = w_sum[w_sum > thres].shape[0]
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def frac_diff_FFD(series, d, thres=1e-5):
    """
    The window terminates after weight goes below a certain threshold
    Eliminates the negative drift of consistently adding weights
    """
    w = get_weights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            # gets the current loc and the loc "width" days back
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            # dot product of weights times the values of this time period
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def demo():
    close = get_tick('AAL')
    # adjust weights to deal with nonstationarity
    # plot_weights([0, 1], 11, size=6)
    # plot_weights([1, 2], 11, size=6)
    print(adfuller(close, 12))
    
    outputs = []
    ds = np.linspace(0, 1, 11)
    for d in ds:
        df1 = np.log(close).resample('1D').last().to_frame()
        df2 = frac_diff(df1, d, thres=.1)
        df2 = adfuller(close, maxlag=1, regression='c', autolag=None)
        # Pvalue
        outputs.append(df2[1])
    plt.plot(ds, outputs)
    plt.savefig(PNG_PATH + "frac_diff.png")
    plt.close()

    outputs = []
    ds = np.linspace(0, 1, 11)
    for d in ds:
        df1 = np.log(close).resample('1D').last().to_frame()
        df2 = frac_diff_FFD(df1, d, thres=.1)
        df2 = adfuller(close, maxlag=1, regression = 'c', autolag=None)
        # Pvalue
        outputs.append(df2[1])
    plt.plot(ds, outputs)
    plt.savefig(PNG_PATH + "frac_diff_FFD.png")
    plt.close()
    

def demo2():
    close = get_tick('AAL')
    x = np.random.randn(close.shape[0])
    dummy = pd.DataFrame({'Close': x.cumsum()}, index=close.index)
    dummy.plot()
    plt.savefig(PNG_PATH + "dummy.png")
    plt.close()
    
    frac_df = frac_diff_FFD(dummy, .5)
    print(frac_df.head())
    
    frac_df = frac_diff_FFD(frac_diff_FFD(dummy, 1), -1)
    print(frac_df.head())

    corrs = []
    ds = []
    # want to keep correlation high, if d is too high correlation goes away and no predictive power
    for d in np.linspace(0, 2, 11):
        frac_df = frac_diff_FFD(dummy, d)
        close_frac = frac_df["Close"]
        corr = close_frac.corr(dummy["Close"])
        print(d, corr)
        if np.isfinite(corr):
            corrs.append(corr)
            ds.append(d)
    plt.plot(ds, corrs)
    # correlation decreases as parameter 'd' is increased
    plt.savefig(PNG_PATH + "frac_df_dummy_corr.png")
    plt.close()
    
    ps = []
    ds = []
    # Also want to get stationarity without removing correlation, value of d around 0.6 is usually a good compromise
    # p value close to 0 around 0.6 with correlation still present
    for d in np.linspace(0, 2, 11):
        frac_df = frac_diff_FFD(dummy, d)
        close_frac = frac_df["Close"]
        close_ = dummy["Close"].loc[close_frac.index]
        if len(close_) > 0:
            # Coefficient of the first argument will change
            res = statsmodels.tsa.stattools.coint(close_frac, close_)
            ps.append(res[1])
            ds.append(d)
    plt.plot(ds, ps)
    # correlation decreases as parameter 'd' is increased
    plt.savefig(PNG_PATH + "frac_df_dummy_p_v_d.png")
    plt.close()
    print(ps)
    
    ps = []
    ds = []
    for d in np.linspace(0, 2, 11):
        frac_df = frac_diff_FFD(dummy, d)
        close_frac = frac_df["Close"]
        if len(close_frac) > 0:
            # Coefficient of the first argument will change
            # goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution
            res = stats.jarque_bera(close_frac)
            ps.append(res[1])
            ds.append(d)
    plt.plot(ds, ps)
    plt.savefig(PNG_PATH + "frac_df_dummy_jarque_bera.png")
    plt.close()
    print(ps)
    

def demo3():
    close = get_tick('AAL')
    frac_df = frac_diff_FFD(close.to_frame(), 0.5)
    vol = get_daily_vol(close)
    events = cusum_filter(close, 2 * vol)
    t1 = get_t1(close, events, num_days=5)
    sampled = get_3barriers(close, events, ptsl=2, trgt=vol, min_ret=0, num_threads=12, t1=t1, side=None)
    data = sampled.dropna()
    print(data)
    features_df = frac_df.loc[data.index].dropna()
    features = features_df.values
    # get the labels of these events
    label = data['t1_type'].loc[features_df.index].values
    clf = RandomForestClassifier()
    # learn on these features and labels
    clf.fit(features, label)
    # predict the features (on the same data so overfitting could be an issue)
    print(clf.predict(features))


if __name__ == '__main__':
    # demo()
    # demo2()
    demo3()