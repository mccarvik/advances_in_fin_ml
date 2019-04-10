import pdb
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from ch7.cross_val_snips import PurgedKFold, cv_score
from util.multiprocessing import mp_pandas_obj
from util.utils import PNG_PATH, DATA_PATH


def plot_feat_importance(path_out, imp, oob, oos, method, tag=0, sim_num=0, **kwargs):
    plt.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'],
                          error_kw={'error': 'r'})
    if method == 'MDI':
        plt.xlim([0,imp.sum(axis = 1).max()])
        plt.axvline(1./imp.shape[0],linewidth = 1,color = 'r',linestyle = 'dotted')
    ax.get_yaxis().set_visible(False)
    plt.savefig(PNG_PATH + "feat_imp_" + method + ".png")
    plt.close()


def test_func(n_features=40, n_informative=10, n_redundant=10, n_estimators=50, n_samples=100, n_splits=3):
    """
    main function to generate data, call feature importance and process output
    """
    X, cont = get_test_data(n_features, n_informative, n_redundant, n_samples)
    config = {'min_w_leaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'],
              'max_samples': [1.]}
    jobs = [dict(zip(config.keys(), conf)) for conf in product(*config.values())]
    kwargs = {'path_out': './test_func/', 'n_estimators': n_estimators,
              'tag': 'test_func', 'n_splits': n_splits}
    out = []
    # one job for each method - SFI, MDA, MDI
    for job in jobs:
        job['sim_num'] = job['method']+'_'+job['scoring']+'_'+'%.2f'%job['min_w_leaf']\
            + '_'+str(job['max_samples'])
        print(job['sim_num'])
        kwargs.update(job)
        imp, oob, oos = feat_importance(X=X, cont=cont, **kwargs)
        plot_feat_importance(imp=imp, oob=oob, oos=oos, **kwargs)
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos})
        df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'min_w_leaf', 'max_samples'])
    out = out[['method','scoring','min_w_leaf','max_samples','I','R','N','oob','oos']]
    out.to_csv(DATA_PATH + 'feat_imp_stats.cvs')


def get_test_data(n_features=40, n_informative=10, n_redundant=10, n_samples=100):
    """
    Function makes test data into classes with features for us to test feature importance
    """
    # Will make randon numbers wuth n_features and put them into classes (default = 2)
    # n_redundant creates duplicatd features (will be removed by analysis)
    X, clss = make_classification(n_samples=n_samples, n_features=n_features,
                                  n_informative=n_informative, n_redundant=n_redundant,
                                  random_state=0, shuffle=False)
    time_idx = pd.DatetimeIndex(periods=n_samples, freq=pd.tseries.offsets.BDay(),
                                end=pd.datetime.today())
    X = pd.DataFrame(X, index=time_idx)
    clss = pd.Series(clss, index=time_idx).to_frame('bin')
    # Create name of columns
    columns = ['I_' + str(i) for i in range(n_informative)]
    columns += ['R_' + str(i) for i in range(n_redundant)]
    columns += ['N_' + str(i) for i in range(n_features - len(columns))]
    X.columns = columns
    # set equal weight
    clss['w'] = 1. / clss.shape[0]
    clss['t1'] = pd.Series(clss.index, index=clss.index)
    return X, clss


def get_e_vec(dot, var_thres):
    """
    If you can draw a line through the three points (0,0), v and Av, then Av is just v multiplied by a number λ; 
    that is, Av=λv. In this case, we call λ an eigenvalue and v an eigenvector. 
    For example, here (1,2) is an eigvector and 5 an eigenvalue.
    Av=(1 8  * (1 = 5 *(1  = λv
        2 1)    2)     2)
    """
    e_val, e_vec = np.linalg.eigh(dot)
    # Descending order
    idx = e_val.argsort()[::-1]
    e_val = e_val[idx]
    e_vec = e_vec[:, idx]
    # Use only positive ones
    e_val = pd.Series(e_val, index=['PC_' + str(i + 1) for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=dot.index, columns=e_val.index)
    e_vec = e_vec.loc[:, e_val > 0]
    e_val = e_val.loc[e_val > 0]
    # Reduce dimension by removing values below threashold
    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.values.searchsorted(var_thres)
    e_val = e_val.iloc[:dim+1]
    e_vec = e_vec.iloc[:, :dim+1]
    return e_val, e_vec


def orth_feats(dfX, var_thres=.95):
    """
    orthogonalization procedure(ex: PCA) alleviates the impact of linear substitution effects
    where one factor explains a similar amount of the decision making as another
    Below is PCA
    """
    # get zscores
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    # dot product of zscores vs itself transposed
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    # get the eigen values and eigen vectors
    e_val, e_vec = get_e_vec(dot, var_thres)
    # dot product of eigenvectors on zscores
    dfP = pd.DataFrame(np.dot(dfZ, e_vec), index=dfZ.index, columns=e_vec.columns)
    return dfP


def feat_imp_MDI(forest, feat_names):
    """
    Mean decrease impurity (MDI) is a fast, explanatory-importance (in-sample, IS) method 
    specific to tree-based classifiers, like Random Forests
    At each node of each decision tree, the selected feature splits the subset it received in such a way 
    that impurity is maximally decreased, can rank features for significance
    - Value is bounded between 0 and 1 and all sum to 1
    - Method does not address substitution effects 
    - cannot be generalized to other non-tree based classifiers
    - set max_features = 1 to avoid masking effects of one feature on another
        - ex: pick on feature that has a lot of overlap with another, so the other feature looks like it has no predictive power
    """
    # uses skleanr built in estimators
    imp_dict = {i:tree.feature_importances_ for i, tree in enumerate(forest.estimators_)}
    imp_df = pd.DataFrame.from_dict(imp_dict, orient='index')
    imp_df.columns = feat_names
    # 0 simply means not used for splitting
    imp_df = imp_df.replace(0, np.nan)
    imp = pd.concat({'mean': imp_df.mean(),
                     'std': imp_df.std() * np.sqrt(imp_df.shape[0])},
                    axis=1)
    imp /= imp['mean'].sum()
    return imp


def feat_imp_MDA(clf, X, y, n_splits, sample_weight, t1, pct_embargo, scoring='neg_log_loss'):
    """
    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method
    1. fits a classifier, 2. derives its performance OOS according to some performance score (accuracy, negativelog-loss, etc.)
    3. permutates each column of the features matrix (X), one column at a time, deriving the performance OOS after each column’s permutation
    The importance of a feature is a function of the loss in performance caused by its column’s permutation
    - can be applied to any classifier
    - not limited to accuracy as the sole performance score
    - susceptible to substitution effects in the presence of correlated features
    - cross validation must be purged and embargoed
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method')
    cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    index = np.arange(n_splits)
    scores = pd.Series(index=index)
    scores_perm = pd.DataFrame(index=index, columns=X.columns)
    for idx, (train, test) in zip(index, cv_gen.split(X=X)):
        print("MDA split index: {}".format(idx))
        # for each split we train a Random Forest
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        w_train = sample_weight.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        w_test = sample_weight.iloc[test]
        clf_fit = clf.fit(X_train, y_train, sample_weight=w_train.values)
        if scoring == 'neg_log_loss':
            prob = clf_fit.predict_proba(X_test)
            scores.loc[idx] = -log_loss(y_test, prob, sample_weight=w_test.values, labels=clf_fit.classes_)
        else:
            pred = clf_fit.predict(X_test)
            scores.loc[idx] = accuracy_score(y_test,  pred, sample_weight=w_test.values)
        
        for col in X.columns:
            print("MDA testing column: {}".format(col))
            X_test_ = X_test.copy(deep=True)
            # Randomize certain feature to make it not effective, then see how much predicatbility is lost
            # higher loss = more predictability value for that parameter
            np.random.shuffle(X_test_[col].values)
            if scoring == 'neg_log_loss':
                prob = clf_fit.predict_proba(X_test_)
                scores_perm.loc[idx, col] = -log_loss(y_test, prob, sample_weight=w_test.value,labels=clf_fit.classes_)
            else:
                pred = clf_fit.predict(X_test_)
                scores_perm.loc[idx, col] = accuracy_score(y_test, pred, sample_weight=w_test.values)
    # (Original score) - (premutated score)
    imprv = (-scores_perm).add(scores, axis=0)
    # Relative to maximum improvement
    if scoring == 'neg_log_loss':
        max_imprv = -scores_perm
    else:
        max_imprv = 1. - scores_perm
    imp = imprv / max_imprv
    imp = pd.DataFrame({'mean': imp.mean(), 'std': imp.std() * np.sqrt(imp.shape[0])})
    return imp, scores.mean()
    

def aux_feat_imp_SFI(feat_names, clf, X, cont, scoring, cv_gen):
    """
    Substitution effects can lead us to discard important features that happen to be redundant
    generally not a problem for prediction, but can lead us to wrong conclusions when we are trying to understand, improve, or simplify a model
    cross-section predictive-importance (out-ofsample, OOS) method
    It computes the OOS performance score of each feature in isolation
    - can be applied to any classifier, not only tree-based classifiers
    - not limited to accuracy as the sole performance score
    - Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time
    """
    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat_name in feat_names:
        # Steps through each of the features and gets a cross validation score with just that feature as x-input
        print("Testing feature: {}".format(feat_name))
        scores = cv_score(clf, X=X[[feat_name]], y=cont['bin'],
                          sample_weight=cont['w'],
                          scoring=scoring,
                          cv_gen=cv_gen)
        imp.loc[feat_name, 'mean'] = scores.mean()
        imp.loc[feat_name, 'std'] = scores.std() * np.sqrt(scores.shape[0])
        print("Finished feature: {}".format(feat_name))
    print("completed aux imp SFI")
    return imp


def feat_importance(X, cont, clf=None, n_estimators=1000, n_splits=10, max_samples=1., num_threads=24, pct_embargo=0., scoring='accuracy', method='SFI', min_w_leaf=0., **kwargs):
    """
    takes in a classifier and type of importance score and runs through the cross validation score
    """
    n_jobs = (-1 if num_threads > 1 else 1)
    # Build classifiers
    if clf is None:
        base_clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                          class_weight='balanced',
                                          min_weight_fraction_leaf=min_w_leaf)
        clf = BaggingClassifier(base_estimator=base_clf, n_estimators=n_estimators,
                                max_features=1., max_samples=max_samples,
                                oob_score=True, n_jobs=n_jobs)
    fit_clf = clf.fit(X, cont['bin'], sample_weight=cont['w'].values)
    if hasattr(fit_clf, 'oob_score_'):
        oob = fit_clf.oob_score_
    else:
        oob = None
    # cv score will use true out of sample training sets
    if method == 'MDI':
        imp = feat_imp_MDI(fit_clf, feat_names=X.columns)
        oos = cv_score(clf, X=X, y=cont['bin'], n_splits=n_splits,
                       sample_weight=cont['w'], t1=cont['t1'],
                       pct_embargo=pct_embargo, scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = feat_imp_MDA(clf, X=X, y=cont['bin'], n_splits=n_splits,
                                sample_weight=cont['w'], t1=cont['t1'],
                                pct_embargo=pct_embargo, scoring=scoring)
    elif method == 'SFI':
        cv_gen = PurgedKFold(n_splits=n_splits, t1=cont['t1'], pct_embargo=pct_embargo)
        oos = cv_score(clf, X=X, y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cv_gen=cv_gen)
        clf.n_jobs = 24
        imp = mp_pandas_obj(aux_feat_imp_SFI, ('feat_names', X.columns),
                            num_threads, clf=clf, X=X, cont=cont,
                            scoring=scoring, cv_gen=cv_gen)
    return imp, oob, oos


def demo():
    X, cont = get_test_data()
    # print(X.head())
    # print(cont.head())
    
    dfP = orth_feats(X)
    # print(dfP.shape)
    # print(dfP.head())
    
    # oob score = out of bag score, the values used as a test set not in the train set for each fold to calculate error
    # i.e. the training dataset was samples not in this "bag"
    # oos score = out of sample score = values not in the actual training set
    clf = RandomForestClassifier(oob_score=True, n_estimators=100) 
    imp_MDI, oob_MDI, oos_MDI = feat_importance(dfP, cont, clf=clf, method='MDI')
    print(imp_MDI.head())
    print(oob_MDI)
    print(oos_MDI)
    print(imp_MDI.sort_values('mean', ascending=False).index)
    
    clf = RandomForestClassifier(oob_score=True, n_estimators=100) 
    imp_MDA, oob_MDA, oos_MDA = feat_importance(dfP, cont, clf=clf, method='MDA')
    print(imp_MDA.head())
    print(oob_MDA)
    print(oos_MDA)
    print(imp_MDA.sort_values('mean', ascending=False).index)
    
    clf = RandomForestClassifier(oob_score=True, n_estimators=100) 
    imp_SFI, oob_SFI, oos_SFI = feat_importance(dfP, cont, clf=clf, method='SFI', n_splits=3)
    print(imp_SFI.head())
    print(oob_SFI)
    print(oos_SFI)
    print(imp_SFI.sort_values('mean', ascending=False).index)


def demo2():
    X, cont = get_test_data()
    dfP = orth_feats(X)

    # concatenates the orthogal features on the original ones
    X_tilde = pd.concat((X, dfP), axis=1)
    print(X_tilde.shape)
    
    clf = RandomForestClassifier(oob_score=True, n_estimators=50) 
    imp_MDI, oob_MDI, oos_MDI = feat_importance(X_tilde, cont, clf=clf, method='MDI', n_splits=3)
    print(imp_MDI.head())
    print(oob_MDI)
    print(oos_MDI)
    
    clf = RandomForestClassifier(oob_score=True, n_estimators=50) 
    imp_MDA, oob_MDA, oos_MDA = feat_importance(X_tilde, cont, clf=clf, method='MDA', n_splits=3)
    print(imp_MDA.head())
    print(oob_MDA)
    print(oos_MDA)
    
    clf = RandomForestClassifier(oob_score=True, n_estimators=50) 
    imp_SFI, oob_SFI, oos_SFI = feat_importance(X_tilde, cont, clf=clf, method='SFI', n_splits=3)
    print(imp_SFI.head())
    print(oob_SFI)
    print(oos_SFI)
    
    print("MDI")
    print(imp_MDI.sort_values('mean', ascending=False).index)
    print("MDA")
    print(imp_MDA.sort_values('mean', ascending=False).index)
    print("SFI")
    print(imp_SFI.sort_values('mean', ascending=False).index)


def demo3():
    test_func()


if __name__ == '__main__':
    # demo()
    # demo2()
    demo3()