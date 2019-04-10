import pdb
import numpy as np
from random import gauss
from itertools import product

def batch(coeffs, n_iter=1e5, max_hp=100,
          prf=np.linspace(.5, 10, 20),
          loss=np.linspace(.5, 10, 20),
          seed=0):
    """
    Simulations based on various parameters:
        sigma --> standard deviation
        forecast --> predicted return on the stock
        half-life --> dictates value of phi i.e. how quickly the process converges to the expected value
        max holding period --> maximum amount of days held
    """
    # phi --> speed at which P0 converges to E[P]
    phi = 2 ** (-1. / coeffs['hl'])
    outputs = []
    for thrs in product(prf, loss):
        # thrs --> eevery combination of profit taking and stop loss strategies from input
        in_outputs = []
        for i in range(int(n_iter)):
            p = seed
            hp = 0
            count = 0
            while True:
                # Ornstein-Uhlenbeck (O-U) process below
                # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                diff = p - seed
                hp += 1
                if diff > thrs[0] or diff < -thrs[1] or hp > max_hp:
                    in_outputs.append(diff)
                    break
        mean = np.mean(in_outputs)
        std = np.std(in_outputs)
        # calculates sharpe ratio for each scenario
        # profit taking, stop loss, mean return, st_dev, sharpe ratio
        print(thrs[0], thrs[1], mean, std, mean/std)
        outputs.append((thrs[0], thrs[1], mean, std, mean/std))
    return outputs

if __name__ == '__main__':
    prf = loss = np.linspace(0, 10, 20)
    count = 0
    # 
    forecasts = [10,5,0,-5,-10]
    # half lifes
    sigmas = [5,10,25,50,100]
    outputs = []
    for item in product(forecasts, sigmas):
        count += 1
        coeffs = {'forecast': item[0], 'hl': item[1], 'sigma': 1}
        output = batch(coeffs, n_iter=1e5, max_hp=100, prf=prf, loss=loss)
        outputs.append(output)