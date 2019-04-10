import pdb
import sys
import time
import timeit
import numpy as np
import pandas as pd
import multiprocessing as mp
import copyreg, types
from itertools import product
sys.path.append("/home/ubuntu/workspace/advances_in_fin_ml")
from data.grab_data import get_tick, get_google_all


def barrierTouch(r, width=.5):
	# find the index of the earliest barrier touch
	t, p = {}, np.log((1 + r).cumprod(axis=0))
	for j in range(r.shape[1]):  # go through columns
		for i in range(r.shape[0]):  # go through rows
			if p[i, j] >= width or p[i, j] <= -width:
				t[j] = i
				continue
	return t


def linParts(numAtoms, numThreads):
    # SNIPPET 20.5 THE linParts FUNCTION
	# partition of atoms with a single loop
	parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
	parts = np.ceil(parts).astype(int)
	return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    # SNIPPET 20.6 THE nestedParts FUNCTION
	# partition of atoms with an inner loop
	parts, numThreads_ = [0], min(numThreads, numAtoms)
	for num in range(numThreads_):
		part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
		part = (-1 + part ** .5) / 2.
		parts.append(part)
	parts = np.round(parts).astype(int)
	if upperTriang:  # the first rows are the heaviest
		parts = np.cumsum(np.diff(parts)[::-1])
		parts = np.append(np.array([0]), parts)
	return parts


# SNIPPET 20.7 THE mpPandasObj, USED AT VARIOUS POINTS IN THE BOOK
# mpEngine.py

# SNIPPET 20.8 SINGLE-THREAD EXECUTION, FOR DEBUGGING
# mpEngine.py

# SNIPPET 20.9 EXAMPLE OF ASYNCHRONOUS CALL TO PYTHON’S MULTIPROCESSING LIBRARY
# mpEngine.py

def expandCall(kargs):
    # SNIPPET 20.10 PASSING THE JOB (MOLECULE) TO THE CALLBACK FUNCTION
	# Expand the arguments of a callback function, kargs[’func’]
	func = kargs['func']
	del kargs['func']
	out = func(**kargs)
	return out


def _pickle_method(method):
    # SNIPPET 20.11 PLACE THIS CODE AT THE BEGINNING OF YOUR ENGINE
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
	for cls in cls.mro():
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)


# SNIPPET 20.12 ENHANCING processJobs TO PERFORM ON-THE-FLY OUTPUT REDUCTION
def processJobsRedux(jobs, task=None, numThreads=24, redux=None, reduxArgs={}, reduxInPlace=False):
	"""
	Run in parallel
	jobs must contain a ’func’ callback, for expandCall
	redux prevents wasting memory by reducing output on the fly
	"""
	if task is None:
		task = jobs[0]['func'].__name__
	pool = mp.Pool(processes=numThreads)
	imap, out, time0 = pool.imap_unordered(expandCall, jobs), None, time.time()
	# Process asynchronous output, report progress
	for i, out_ in enumerate(imap, 1):
		if out is None:
			if redux is None:
				out, redux, reduxInPlace = [out_], list.append, True
			else:
				out = copy.deepcopy(out_)
		else:
			if reduxInPlace:
				redux(out, out_, **reduxArgs)
			else:
				out = redux(out, out_, **reduxArgs)
		reportProgress(i, len(jobs), time0, task)
	pool.close()
	pool.join()  # this is needed to prevent memory leaks
	if isinstance(out, (pd.Series, pd.DataFrame)):
		out = out.sort_index()
	return out


def mpJobList(func, argList, numThreads=24, mpBatches=1, linMols=True,
		    redux=None, reduxArgs={}, reduxInPlace=False, **kargs):
    # SNIPPET 20.13 ENHANCING mpPandasObj TO PERFORM ON-THE-FLY OUTPUT REDUCTION		    
	if linMols:
		parts = linParts(len(argList[1]), numThreads * mpBatches)
	else:
		parts = nestedParts(len(argList[1]), numThreads * mpBatches)
	jobs = []
	for i in xrange(1, len(parts)):
		job = {argList[0]: argList[1][parts[i - 1]:parts[i]], 'func': func}
		job.update(kargs)
		jobs.append(job)
	out = processJobsRedux(jobs, redux=redux, reduxArgs=reduxArgs, reduxInPlace=reduxInPlace, numThreads=numThreads)
	return out


# ——————————————————————————————————————
def getPCs(path, molecules, eVec):
	# get principal components by loading one file at a time
	pcs = None
	df0 = pd.DataFrame()
	for i in molecules:
		df0 = pd.read_csv(path + i, index_col=0, parse_dates=True)
		if pcs is None:
			pcs = np.dot(df0.values, eVec.loc[df0.columns].values)
		else:
			pcs += np.dot(df0.values, eVec.loc[df0.columns].values)
	pcs = pd.DataFrame(pcs, index=df0.index, columns=eVec.columns)
	return pcs


def reportProgress(jobNum,numJobs,time0,task):
    # SNIPPET 20.9 EXAMPLE OF ASYNCHRONOUS CALL TO PYTHON’S MULTIPROCESSING LIBRARY
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs,(time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
    str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return


def demo():
    # SNIPPET 20.1 UN-VECTORIZED CARTESIAN PRODUCT
	# Cartesian product of dictionary of lists
	dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}
	for a in dict0['a']:
		for b in dict0['b']:
			for c in dict0['c']:
				print({'a': a, 'b': b, 'c': c})


def demo2():
    # SNIPPET 20.2 VECTORIZED CARTESIAN PRODUCT
	# Cartesian product of dictionary of lists
	dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}
	jobs = (dict(zip(dict0, i)) for i in product(*dict0.values()))
	for i in jobs:
		print(i)


def demo3():
    # SNIPPET 20.3 SINGLE-THREAD IMPLEMENTATION OF A ONE-TOUCH DOUBLE BARRIER
	# Path dependency: Sequential implementation
	r = np.random.normal(0, .01, size=(1000, 10000))
	t = barrierTouch(r)
	print(t)


def demo4():
    func = demo3
    print(min(timeit.Timer(func).repeat(5, 10)))


def demo5():
    # SNIPPET 20.4 MULTIPROCESSING IMPLEMENTATION OF A ONE-TOUCH DOUBLE BARRIER
    # Path dependency: Multi-threaded implementation
	r, numThreads = np.random.normal(0, .01, size=(1000, 10000)), 24
	parts = np.linspace(0, r.shape[0], min(numThreads, r.shape[0]) + 1)
	parts, jobs = np.ceil(parts).astype(int), []
	for i in range(1, len(parts)):
		jobs.append(r[:, parts[i - 1]: parts[i]])  # parallel jobs
	pool, out = mp.Pool(processes=numThreads), []
	outputs = pool.imap_unordered(barrierTouch, jobs)
	for out_ in outputs:
		out.append(out_)  # asynchronous response
	pool.close()
	pool.join()
	return


def demo6():
    a = copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    print(a)


def demo7():
    # SNIPPET 20.14 PRINCIPAL COMPONENTS FOR A SUBSET OF THE COLUMNS
    pcs = mpJobList(getPCs, ('molecules', fileNames), numThreads=24, mpBatches=1,
					path=path, eVec=eVec, redux=pd.DataFrame.add)


if __name__ == '__main__':
    # demo()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    # demo6()
    demo7()