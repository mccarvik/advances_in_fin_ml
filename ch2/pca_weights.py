import numpy as np

def pcaWeights(cov, riskDist=None, riskTarget=1.):
    """
    Calculate PCA weights 
    """
    # Following the riskAlloc distribution, match riskTarget
    eVal, eVec = np.linalg.eigh(cov) # must be Hermitian
    indices = eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal, eVec = eVal[index], eVec[:indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1
    loads = riskTarget * (riskDist / eVal) ** .5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # ctr = (loads / riskTarget)**2 * eVal # verify riskDits
    return wghts
        