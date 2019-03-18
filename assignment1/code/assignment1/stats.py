from collections import namedtuple
import functools
import itertools
from multiprocessing import Pool
import re
import warnings

from scipy import stats


__all__ = ['distribution_fit', 'DistributionFit']

DistributionFit = namedtuple('DistributionFit', ['name', 'args', 'fit'])

def distribution_fit(data):
    distributions = ('alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'gennorm', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa4', 'kappa3', 'ksone', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 'skewnorm', 't', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy')
    results = []
    with Pool(initializer=_warning_disabler) as p:
        results.extend(p.map(functools.partial(_distribution_fit, data), distributions))
        # results.extend(p.map(DistributionFitter(data), distributions))
    print(f"Could not proccess {results.count(None)} distributions")
    return tuple(sorted((result for result in results if result is not None),
                        key=lambda dist: dist.fit))


class DistributionFitter:
    def __init__(self, data):
        self.data = data

    def __call__(self, distribution):
        return _distribution_fit(self.data, distribution)

def _warning_disabler():
    warnings.simplefilter("ignore")

def _distribution_fit(data, distribution):
    fit = args = ()
    try:
        fit = stats.kstest(data, distribution)
    except TypeError as e:
        missing_arguments = int(re.search('\d+', str(e)).group(0))
        if 1 <= missing_arguments <= 2:
            available_args = (range(1,11),)*missing_arguments

            best_fit = ()
            for arg in itertools.product(*available_args):
                fit = stats.kstest(data, distribution, args=arg)
                if not best_fit or fit < best_fit:
                    best_fit = fit
                    args = (arg,)
            fit = best_fit
        else:
            print(distribution, missing_arguments)

    return DistributionFit(distribution, args, fit) if fit else None
