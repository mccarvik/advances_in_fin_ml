import pdb
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_google, get_tick, get_google_all
from ch3.labeling_snips import cusum_filter, get_t1, get_3barriers, get_daily_vol
from ch4.sample_wgt_snips import get_num_co_events, get_sample_tw


def get_embargo_times(times, pct_embargo):
    """
    reduce leakage by purging from the training set all observations whose labels overlapped 
    in time with those labels included in the testing set
    """
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        embg_times = pd.Series(times, index=times)
    else:
        embg_times = pd.Series(times[step:], index=times[:-step])
        embg_times = embg_times.append(pd.Series(times[-1], index=times[:-step]))
    return embg_times


def get_train_times(t1, test_times):
    trn = t1.copy(deep=True)
    for i, j in test_times.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        df1 = trn[(i <= trn) & (trn <= j)].index
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        trn = trn.drop(df0.union(df1.union(df2)))
    return trn


def cv_score(clf, X, y, sample_weight=None, scoring='neg_log_loss', t1=None, n_splits=3, cv_gen=None, pct_embargo=0., purging=False):
    """
    Simple cross validation score, retursn array of scores for each fold
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('Wrong scoring method')
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1,
                             pct_embargo=pct_embargo,
                             purging=purging)
    scores = []
    # cv_gen.split() returns the start and end point of folds
    for train, test in cv_gen.split(X=X):
        train_params = dict()
        test_params = dict()
        # Sample weight is an optional parameter
        if sample_weight is not None:
            train_params['sample_weight'] = sample_weight.iloc[train].values
            test_params['sample_weight'] = sample_weight.iloc[test].values
        clf_ = clf.fit(X=X.iloc[train, :], y=y.iloc[train], **train_params)
        # Scoring
        if scoring == 'neg_log_loss':
            prob = clf_.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, labels=clf.classes_, **test_params)
        else:
            pred = clf_.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, **test_params)
        scores.append(score_)
    return np.array(scores)


class PurgedKFold(_BaseKFold):
    """
    KFold cross validation with purging of data based on embargo pct
    """
    def __init__(self, n_splits=3, t1=None, pct_embargo=0., purging=True):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label through dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits=n_splits, shuffle=False,
                                          random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.purging = purging
        
    def split(self, X, y=None, groups=None):
        """
        Splits up data set into train and test via n_splits parameter
        Yields a different fold for each call
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and t1 must have the same index')
        indices = np.arange(X.shape[0])
        # Embargo width
        embg_size = int(X.shape[0] * self.pct_embargo)
        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        for st, end in test_ranges:
            # Test data
            test_indices = indices[st:end]
            # Training data prior to test data
            t0 = self.t1.index[st]
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # Add training data after test data
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + embg_size:]))
            # Purging
            if self.purging:
                train_t1 = self.t1.iloc[train_indices]
                test_t1 = self.t1.iloc[test_indices]
                train_t1 = get_train_times(train_t1, test_t1)
                train_indices = self.t1.index.searchsorted(train_t1.index)
            yield train_indices, test_indices


def demo():
    # close = get_tick('AAL')
    df = get_google_all()
    df.index = pd.DatetimeIndex(df['Date'].values)
    close = df["Close"]
    embg_times = get_embargo_times(close.index, pct_embargo=0.01)
    print(embg_times.head())
    
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=1)
    side =  None
    # events = get_3barriers(close, t_events=sampled_idx, trgt=vol,ptsl=[1, 2], t1=t1, side=side)
    events = get_3barriers(close, t_events=sampled_idx, trgt=vol,ptsl=1, t1=t1, side=side)
    print(events.head())
    
    index = events.index
    features_df = df.drop(columns=["Date"]).dropna().loc[index]
    features = features_df
    label = events['t1_type'].loc[features_df.index]

    # without shuffling
    scores = []
    for _ in range(10):   
        clf = RandomForestClassifier()
        kfold = KFold(n_splits=10, shuffle=False)
        scores.append(cross_val_score(clf, features, label, cv=kfold))
    print(np.mean(scores), np.var(scores))
    
    # with shuffling the data before putting into batches
    # Shffuling data introduces data leakage because of simlarity among neighborg, 
    # If you shuffle data uniformly, training data has more information that overlaps test data.
    scores = []
    for _ in range(10):   
        clf = RandomForestClassifier()
        kfold = KFold(n_splits=10, shuffle=True)
        scores.append(cross_val_score(clf, features, label, cv=kfold))
    print(np.mean(scores), np.var(scores))


def demo2():
    df = get_google_all()
    df.index = pd.DatetimeIndex(df['Date'].values)
    close = df["Close"]
    
    vol = get_daily_vol(close)
    sampled_idx = cusum_filter(close, vol)
    t1 = get_t1(close, sampled_idx, num_days=1)
    side =  None
    events = get_3barriers(close, t_events=sampled_idx, trgt=vol,ptsl=1, t1=t1, side=side)
    index = events.index
    features_df = df.drop(columns=["Date"]).dropna().loc[index]
    features = features_df
    label = events['t1_type'].loc[features_df.index]
    
    clf = RandomForestClassifier()
    t1_ = t1.loc[features.index]
    
    # No purge, with embargo
    scores = []
    for _ in range(10):
        scores_ = cv_score(clf, features, label, pct_embargo=0.01, t1=t1_, purging=False)
        scores.append(np.mean(scores_))
    print(np.mean(scores), np.var(scores))
    
    # no purge without embargo
    scores = []
    for _ in range(10):
        scores_ = cv_score(clf, features, label, pct_embargo=0., t1=t1_, purging=False)
        scores.append(np.mean(scores_))
    print(np.mean(scores), np.var(scores))
    
    n_co_events = get_num_co_events(close.index, t1, events.index)
    sample_weight = get_sample_tw(t1, n_co_events, events.index)
    
    # no purge with embargo and sample weights added to samples
    scores = []
    for _ in range(10):
        scores_ = cv_score(clf, features, label, sample_weight=sample_weight, pct_embargo=0.01, t1=t1_, purging=False)
        scores.append(np.mean(scores_))
    print(np.mean(scores), np.var(scores))
    
    # no purge without embargo and sample weights added to samples
    scores = []
    for _ in range(10):
        scores_ = cv_score(clf, features, label, sample_weight=sample_weight, pct_embargo=0., t1=t1_, purging=False)
        scores.append(np.mean(scores_))
    print(np.mean(scores), np.var(scores))
    

if __name__ == '__main__':
    # demo()
    demo2()