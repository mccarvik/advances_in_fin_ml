import pdb
import sys
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_tick
from util.utils import PNG_PATH, DATA_PATH
from CLA import CLA


# SNIPPET 16.4 FULL IMPLEMENTATION OF THE HRP ALGORITHM
#---
def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp
    
    
#---
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    # get the inverse variance portfolio and use for the weights on the cluster
    w_=getIVP(cov_).reshape(-1,1)
    # get the current variance of the cluster through the dotproduct
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar
    
    
#---
def getQuasiDiag(link):
    # SNIPPET 16.2 QUASI-DIAGONALIZATION
	# Sort clustered items by distance
    # reorganizes the rows and columns of the covariance matrix, so that the largest values lie along the diagonal
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index
        j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


#---
def getRecBipart(cov,sortIx):
    # SNIPPET 16.3 RECURSIVE BISECTION
    # Compute HRP alloc
    # creates the bifurcation tree
    # set weights
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        # splits list in two
        cItems=[i[int(j):int(k)] for i in cItems for j,k in ((0,len(i)/2), (len(i)/2,len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            # dicatates how much weight to give to each cluster
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist


def plotCorrMatrix(path,corr,labels=None):
    # Heatmap of the correlation matrix
    if labels is None:labels=[]
    plt.pcolor(corr)
    plt.colorbar()
    plt.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    plt.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    plt.savefig(PNG_PATH + path)
    plt.clf(); plt.close() # reset pylab
    return


def generateData(nObs,size0,size1,sigma1):
    # Time series of correlated variables
    #1) generating some uncorrelated data
    np.random.seed(seed=12345);random.seed(12345)
    x=np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
    #2) creating correlation between the variables
    cols=[random.randint(0,size0-1) for i in range(size1)]
    y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
    x=np.append(x,y,axis=1)
    x=pd.DataFrame(x,columns=range(1,x.shape[1]+1))
    return x,cols
    
    
def generateData2(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
	# Time series of correlated variables
	# 1) generate random uncorrelated data
	x = np.random.normal(mu0, sigma0, size=(nObs, size0))
	# 2) create correlation between the variables
	cols = [random.randint(0, size0 - 1) for i in range(size1)]
	y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
	x = np.append(x, y, axis=1)
	# 3) add common random shock
	point = np.random.randint(sLength, nObs - 1, size=2)
	x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
	# 4) add specific random shock
	point = np.random.randint(sLength, nObs - 1, size=2)
	x[point, cols[-1]] = np.array([-.5, 2])
	return x, cols


# ---
def getHRP(cov, corr):
	# Construct a hierarchical portfolio
	corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
	dist = correlDist(corr)
	link = sch.linkage(dist, 'single')
	sortIx = getQuasiDiag(link)
	sortIx = corr.index[sortIx].tolist()  # recover labels
	hrp = getRecBipart(cov, sortIx)
    # get the weights of the different clusters 
	return hrp.sort_index()


# ---
def getCLA(cov, **kargs):
	# Compute CLA's minimum variance portfolio
    # Uses CLA class no idea what the hell is going on in there
	mean = np.arange(cov.shape[0]).reshape(-1, 1)  # Not used by C portf
	lB = np.zeros(mean.shape)
	uB = np.ones(mean.shape)
	cla = CLA(mean, cov, lB, uB)
	cla.solve()
	return cla.w[-1].flatten()


def demo():
    # SNIPPET 16.1 TREE CLUSTERING USING SCIPY FUNCTIONALITY
    close = pd.DataFrame()
    for ind_t in ['AAL', 'MSFT', 'CSCO', 'AAPL']:
        if close.empty:
            close = get_tick(ind_t).to_frame()
            close.columns = [ind_t]
        else:
            t_close = get_tick(ind_t).to_frame()
            t_close.columns = [ind_t]
            close = pd.merge(close,t_close, how='inner', left_index=True, right_index=True)
    cov, corr = close.cov(), close.corr()
    print(corr)
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    print(dist)
    # linkage matrix N-1 x 4 matrix
    # y1, y2 report the constituents, y3 reports th distance between y1 and y2, 
    # y4 is number of items in the cluster
    link = sch.linkage(dist, 'single')  # linkage matrix
    print(link)

    quasi_diag = getQuasiDiag(link)
    pdb.set_trace()
    print(quasi_diag)
    rec_bisec = getRecBipart(cov, quasi_diag)
    print(rec_bisec)


def demo2():
    # SNIPPET 16.4 FULL IMPLEMENTATION OF THE HRP ALGORITHM
	# 1) Generate correlated data
	nObs, size0, size1, sigma1 = 10000, 5, 5, .25
	x, cols = generateData(nObs, size0, size1, sigma1)
	print([(j + 1, size0 + i) for i, j in enumerate(cols, 1)])
	cov, corr = x.cov(), x.corr()
	# 2) compute and plot correl matrix
	plotCorrMatrix('HRP3_corr0.png', corr, labels=corr.columns)
	# 3) cluster
	dist = correlDist(corr)
    # linkage matrix
    # linkage matrix N-1 x 4 matrix
    # y1, y2 report the constituents, y3 reports th distance between y1 and y2, 
    # y4 is number of items in the cluster
	link = sch.linkage(dist, 'single')
	sortIx = getQuasiDiag(link)
	sortIx = corr.index[sortIx].tolist()  # recover labels
    # reorders the clusers based on correlation
	df0 = corr.loc[sortIx, sortIx]  # reorder
    # notice the high correlation on the diagonal from the quasi diagonal move
	plotCorrMatrix('HRP3_corr1.png', df0, labels=df0.columns)
	# 4) Capital allocation
	hrp = getRecBipart(cov, sortIx)
	print(hrp)
	return hrp
	

def demo3(numIters=100, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2, sigma1F=.25, sLength=260, rebal=22):
    # Monte Carlo experiment on HRP
    # methods = [getIVP, getHRP, getCLA]
    methods = [getIVP, getHRP]
    stats, numIter = {i.__name__: pd.Series() for i in methods}, 0
    pointers = range(sLength, nObs, rebal)
    while numIter < numIters:
        print(numIter)
        # 1) Prepare data for one experiment - monte carlo data
        x, cols = generateData2(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
        r = {i.__name__: pd.Series() for i in methods}
        # 2) Compute portfolios in-sample
        for pointer in pointers:
            x_ = x[pointer - sLength: pointer]
            cov_, corr_ = np.cov(x_, rowvar=False), np.corrcoef(x_, rowvar=False)
            # 3) Compute performance out-of-sample
            x_ = x[pointer: pointer + rebal]
        for func in methods:
            cov, corr = np.cov(x, rowvar=False), np.corrcoef(x, rowvar=False)
            w_ = func(cov=cov, corr=corr)  # callback
            r_ = pd.Series(np.dot(x, w_))
            r[func.__name__] = r[func.__name__].append(r_)
        # 4) Evaluate and store results
        for func in methods:
            r_ = r[func.__name__].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[func.__name__].loc[numIter] = p_.iloc[-1] - 1
        numIter += 1
    
    # 5) Report results
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    stats.to_csv(DATA_PATH +'stats.csv')
    df0, df1 = stats.std(), stats.var()
    print(pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1))
    return
    

if __name__ == '__main__':
    # demo()
    # demo2()
    demo3()