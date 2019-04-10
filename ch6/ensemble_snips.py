import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def demo():
    clf = DecisionTreeClassifier(criterion='entropy', max_features='auto', class_weight='balanced')
    avg_u = 1.
    bc = BaggingClassifier(base_estimator=clf, n_estimators=1000, max_samples=avg_u, max_features=1.)
    # help(BaggingClassifier)
    
    avg_u = 0.5
    clf0 = RandomForestClassifier(n_estimators=1000, class_weight='balaned_subsample', criterion='entropy')
    clf1 = DecisionTreeClassifier(criterion='entropy', max_features='auto', class_weight='balanced')
    clf1 = BaggingClassifier(base_estimator=clf1, n_estimators=1000, max_samples=avg_u)
    clf2 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
    clf2 = BaggingClassifier(base_estimator=clf2, n_estimators=1000, max_samples=avg_u, max_features=1.)
    # help(DecisionTreeClassifier)
    

if __name__ == '__main__':
    demo()