import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ch8.get_test_data import get_test_data
from finance_ml.datasets import get_cls_data

class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)
        

def clf_hyper_fit(feat, label, t1, pipe_clf, search_params, scoring=None,
                  n_splits=3, bagging=[0, None, 1.],
                  rnd_search_iter=0, n_jobs=-1, pct_embargo=0., **fit_params):
    # Set defaut value for scoring
    if scoring is None:
        if set(label.values) == {0, 1}:
            scoring = 'f1'
        else:
            scoring = 'neg_log_loss'
    # HP serach on traing data
    inner_cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    if rnd_search_iter == 0:
        search = GridSearchCV(estimator=pipe_clf, param_grid=search_params,
                              scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    else:
        search = RandomizedSearchCV(estimator=pipe_clf, param_distributions=search_params,
                                    scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    best_pipe = search.fit(feat, label, **fit_params).best_estimator_
    # Fit validated model on the entirely of dawta
    if bagging[0] > 0:
        bag_est = BaggingClassifier(base_estimator=MyPipeline(best_pipe.steps),
                                   n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                                   max_features=float(bagging[2]), n_jobs=n_jobs)
        bag_est = best_pipe.fit(feat, label,
                                sample_weight=fit_params[bag_est.base_estimator.steps[-1][0] + '__sample_weight'])
        best_pipe = Pipeline([('bag', bag_est)])
    return best_pipe


class LogUniformGen(rv_continuous):
    """
    class for parameters where it is more effective to draw values from a distribution where
    the logarithm of those draws will be distributed uniformly
    """
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)
    
    
def log_uniform(a=1, b=np.exp(1)):
    """
    creates a log uniform distribution
    """
    return LogUniformGen(a=a, b=b, name='log_uniform')
    

def demo():
    a = 1e-3
    b = 1e3
    size = 10000
    vals = log_uniform(a=a, b=b).rvs(size=size)
    print(vals.shape)
    print(vals)
    
    X, label = get_cls_data(n_features=10, n_informative=5, n_redundant=0, n_samples=10000)
    print(X.head())
    print(label.head())
    
    name = 'svc'
    params_grid = {name + '__C': [1e-2, 1e-1, 1, 10, 100], name + '__gamma': [1e-2, 1e-1, 1, 10, 100]}
    kernel = 'rbf'
    clf = SVC(kernel=kernel, probability=True)
    pipe_clf = Pipeline([(name, clf)])
    fit_params = dict()
    clf = clf_hyper_fit(X, label['bin'], t1=label['t1'], pipe_clf=pipe_clf, scoring='neg_log_loss',
                        search_params=params_grid, n_splits=3, bagging=[0, None, 1.],
                        rnd_search_iter=0, n_jobs=-1, pct_embargo=0., **fit_params)
                        
    name = 'svc'
    params_dist = {name + '__C': log_uniform(a=1e-2, b=1e2),
                   name + '__gamma': log_uniform(a=1e-2, b=1e2)}
    kernel = 'rbf'
    clf = SVC(kernel=kernel, probability=True)
    pipe_clf = Pipeline([(name, clf)])
    fit_params = dict()
    
    clf = clf_hyper_fit(X, label['bin'], t1=label['t1'], pipe_clf=pipe_clf, scoring='neg_log_loss',
                        search_params=params_grid, n_splits=3, bagging=[0, None, 1.],
                        rnd_search_iter=25, n_jobs=-1, pct_embargo=0., **fit_params)


if __name__ == '__main__':
    demo()