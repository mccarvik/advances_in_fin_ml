import pdb
import sys
import numpy as np

# ---
def plugIn(msg, w):
    # SNIPPET 18.1 PLUG-IN ENTROPY ESTIMATOR
	# Compute plug-in (ML) entropy rate
    # maximum likelihood entropy estimator
	pmf = pmf1(msg, w)
    # calculates the amount of entropy
	out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
	return out, pmf


# ———————————————————————————————————————
def pmf1(msg, w):
	# Compute the prob mass function for a one-dim discrete rv
	# len(msg)-w occurrences
    # calculates the probability of a substring of length w
	lib = {}
	if not isinstance(msg, str):
		msg = ''.join(map(str, msg))
	for i in range(w, len(msg)):
		msg_ = msg[i - w:i]
		if msg_ not in lib:
			lib[msg_] = [i - w]
		else:
			lib[msg_] = lib[msg_] + [i - w]
	pmf = float(len(msg) - w)
	pmf = {i: len(lib[i]) / pmf for i in lib}
	return pmf


def lempelZiv_lib(msg):
    # SNIPPET 18.2 A LIBRARY BUILT USING THE LZ ALGORITHM
    # Entropy can be interpreted as a measure of complexity
    # Complex sequence contains more information than a regular (predictable) sequence
    i, j, lib = 1, 0, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i: j + 1]
            if msg_ not in lib:
                lib.append(msg_)
            break
        i = j + 1
    return lib


# SNIPPET 18.3 FUNCTION THAT COMPUTES THE LENGTH OF THE LONGEST MATCH
def matchLength(msg, i, n):
    # determines the length of the longest match
	# Maximum matched length+1, with overlap.
	# i>=n & len(msg)>=i+n
	subS = ''
	for l in range(n):
		msg1 = msg[i:i + l + 1]
		for j in range(i - n, i):
			msg0 = msg[j:j + l + 1]
			if msg1 == msg0:
				subS = msg1
				break  # search for higher l.
	return len(subS) + 1, subS  # matched length + 1


# SNIPPET 18.4 IMPLEMENTATION OF ALGORITHMS DISCUSSED IN GAO ET AL. [2008]
def konto(msg, window=None):
	"""
	* Kontoyiannis’ LZ entropy estimate, 2013 version (centered window).
	* Inverse of the avg length of the shortest non-redundant substring.
	* If non-redundant substrings are short, the text is highly entropic.
	* window==None for expanding window, in which case len(msg)%2==0
	* If the end of msg is more relevant, try konto(msg[::-1])
	"""
	out = {'num': 0, 'sum': 0, 'subS': []}
	if not isinstance(msg, str):
		msg = ''.join(map(str, msg))
	if window is None:
		points = range(1, int(len(msg) / 2 + 1))
	else:
		window = min(window, len(msg) / 2)
		points = range(window, len(msg) - window + 1)
	for i in points:
		if window is None:
			l, msg_ = matchLength(msg, i, i)
			out['sum'] += np.log2(i + 1) / l  # to avoid Doeblin condition
		else:
			l, msg_ = matchLength(msg, i, window)
			out['sum'] += np.log2(window + 1) / l  # to avoid Doeblin condition
		out['subS'].append(msg_)
		out['num'] += 1
	out['h'] = out['sum'] / out['num']
	out['r'] = 1 - out['h'] / np.log2(len(msg))  # redundancy, 0<=r<=1
	return out


def demo():
    print(plugIn("hippopotamus", 2))
    # creates lib of unique items
    print(lempelZiv_lib("hippopotamus"))
    print(matchLength("hippopotamus", 3,10))
    
    # SNIPPET 18.4
    msg = '1011101'
    print(konto(msg * 2))
    print(konto(msg + msg[::-1]))


if __name__ == '__main__':
    demo()