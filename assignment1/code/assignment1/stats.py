from collections import namedtuple
import functools
import itertools
from multiprocessing import Pool as ProcessPool
from multiprocessing.pool import ThreadPool
import os
import re
import warnings

import numpy as np
from scipy import stats


__all__ = ['distribution_fit', 'DistributionFit']

DistributionFit = namedtuple('DistributionFit', ['name', 'args', 'fit'])

_distributions = ('alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'gennorm', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa4', 'kappa3', 'ksone', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 'skewnorm', 't', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy')

def np_cache(**kwargs):
    def decorator(function):
        @functools.lru_cache(**kwargs)
        def cached_wrapper(hashable_array):
            array = np.array(hashable_array)
            return function(array)

        @functools.wraps(function)
        def wrapper(array):
            return cached_wrapper(tuple(array))

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper
    return decorator

@np_cache(maxsize=16)
def distribution_fit(data):
    results = []
    with ThreadPool(processes=int(0.25*os.cpu_count()),
                    initializer=_warning_disabler) as p:
        results.extend(p.map(functools.partial(_distribution_fit, data), _distributions))
    return tuple(sorted((result for result in results if result is not None),
                        key=lambda dist: dist.fit))

def _warning_disabler():
    warnings.simplefilter("ignore")

def _kstest(data, distribution, arg):
    return (stats.kstest(data, distribution, args=arg), arg)

def _distribution_fit(data, distribution):
    fit = args = ()
    try:
        fit = stats.kstest(data, distribution)
    except TypeError as e:
        missing_arguments = int(re.search('\d+', str(e)).group(0))
        if 1 <= missing_arguments <= 2:
            available_args = (range(1,11),)*missing_arguments

            with ProcessPool(processes=int(0.5*os.cpu_count()),
                             initializer=_warning_disabler) as p:
                (fit, args) = min(p.map(functools.partial(_kstest, data, distribution), itertools.product(*available_args)))

    return DistributionFit(distribution, args, fit) if fit else None
