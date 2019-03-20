import math

import matplotlib.pylab as plt
import numpy as np
from scipy import stats
from scipy.io import loadmat

from stats import distribution_fit

def _load_signal(name):
    try:
        mat = loadmat(name)
    except FileNotFoundError:
        raise
    return {
            key: value.flatten()
            for (key, value) in mat.items() if not key.startswith('__')
            }

def movmean(signal, w=3, iterations_left=None, direction=0):
    if len(signal) == 0 or w == 0 or iterations_left == 0:
        return []
    elif len(signal) < w:
        return movmean(signal, w-1)

    tmp_sig = np.convolve(signal, np.ones(w), 'valid')/w
    head, tail = signal[:w-1], signal[-w+1:]
    if direction < 0:
        return np.hstack((
                          movmean(head, w-1, iterations_left-1, -1),
                          tmp_sig
                          ))
    elif direction == 0:
        return np.hstack((
                         movmean(head, w-1, math.ceil((w-1)/2.), -1),
                         tmp_sig,
                         movmean(tail, w-1, math.floor((w-1)/2.), +1)
                         ))
    elif direction > 0:
        return np.hstack((
                          tmp_sig,
                          movmean(tail, w-1, iterations_left-1, +1)
                         ))

def error(x, y):
    return np.linalg.norm(x-y)/np.linalg.norm(x)

def move_mean_to_0(signal, x):
    return signal-np.mean(signal)+np.mean(signal)*np.sin(2*np.pi*x)


if __name__ == '__main__':
    P_0, d_0 = 0, 5

    signal_data = _load_signal('prx')
    distance = signal_data['dPath']
    logdistance = np.log10(signal_data['dPath']/d_0)
    prx = signal_data['Prx']
    expected_pathloss = signal_data['pathLoss']
    expected_shading = signal_data['shadCorr']


    # Step 1: Characterizing the pathloss
    m, c = np.linalg.lstsq(
             np.vstack([logdistance,
                        P_0 * np.ones(len(logdistance))
                        ]).T,
             prx,
             rcond=None)[0]

    # The intercept should be equal to P_0, which is certainly
    # guaranteed by the linear system. Adding an assert in order
    # to make it explicit.
    # XXX: Note it may fail because of float point precision, it
    # is not an issue for P_0 = 0 though.
    assert c == P_0

    n = -m/10
    pathloss = -m*logdistance

    print((
         'Pathloss information\n'
         '--------------------\n'
        f'Pathloss coeficient: {n}\n'
        f'Pathloss quadratic error: {error(-expected_pathloss, m*logdistance)}\n'
        ))

    large_scale_fading = movmean(prx, 100)
    small_scale_fading = prx - large_scale_fading
    shading = move_mean_to_0(large_scale_fading+pathloss, distance)


    plt.figure()
    plt.title('Potência recebida e perdas do sinal original')
    plt.plot(distance, prx, 'orange',
            label='Potência recebida completa')
    plt.plot(distance, -expected_pathloss+expected_shading, 'cyan',
            label='Desvanecimento em larga escala')
    plt.plot(distance, -expected_pathloss, 'r--',
             label='Perda de percurso')

    plt.xlabel('Distância (m)')
    plt.ylabel('Potência (dB)')
    plt.xlim(min(distance), max(distance))
    plt.legend()

    plt.savefig('trial1_original.eps')


    plt.figure()
    plt.title('Potência recebida e perdas do sinal estimado')
    plt.plot(distance, prx, 'orange',
             label='Potência recebida completa')
    plt.plot(distance, large_scale_fading, 'cyan',
             label='Desvanecimento em larga escala')
    plt.plot(distance, -pathloss, 'r--',
             label='Perda de percurso')
  
    plt.xlabel('Distância (m)')
    plt.ylabel('Potência (dB)')
    plt.xlim(min(distance), max(distance))
    plt.legend()

    plt.savefig('trial1_estimated.eps')


    plt.figure()
    plt.title('Perda de percurso original e estimada')
    plt.plot(distance, expected_pathloss, 'cyan',
             label='Perda de percurso original')
    plt.plot(distance, pathloss, 'r--',
             label='Perda de percurso estimada')
  
    plt.xlabel('Distância (m)')
    plt.ylabel('Potência (dB)')
    plt.xlim(min(distance), max(distance))
    plt.legend()

    plt.savefig('trial1_pathloss.eps')


    plt.figure()
    plt.title('Sombreamento original e estimado')
    plt.plot(distance, expected_shading, 'cyan',
             label='Sombreamento original')
    plt.plot(distance, shading, 'r--',
             label='Sombreamento estimado')
 
    plt.xlabel('Distância (m)')
    plt.ylabel('Potência (dB)')
    plt.xlim(min(distance), max(distance))
    plt.legend()

    plt.savefig('trial1_shading.eps')

    # distributions = ('anglit', 'arcsine', 'cauchy', 'cosine', 'expon', 'gilbrat', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'kstwobign', 'laplace', 'levy', 'levy_l', 'logistic', 'maxwell', 'moyal', 'norm', 'rayleigh', 'semicircular', 'uniform', 'wald')
    #min(distributions, key=lambda name: stats.kstest(_shading, name))
    # cauchy vs anglit
    # _distributions = sorted(distributions, key=lambda name: stats.kstest(shading, name))
    """plt.figure()
    x = np.linspace(-30,30,1000)
    param = stats.dgamma.fit(shading, 5)
    pdf_fitted = stats.dgamma.pdf(x, param[0], loc=param[1], scale=param[2])
    plt.hist(shading, normed=True, label='Histograma do sombreamento')
    plt.plot(x,pdf_fitted,'r-', label='Fit da distribuição gamma')
    param = stats.cauchy.fit(shading)
    pdf_fitted = stats.cauchy.pdf(x, loc=param[0], scale=param[1])
    plt.plot(x,pdf_fitted,'b-', label='Fit da distribuição de Cauchy')
    plt.legend()
    plt.show()"""
    print(
            'Janela | Desvio padrão | Média | Erro Médio'
        )
    for w in (10, 50, 100, 150, 200):
        _large_scale_fading = movmean(prx, w)
        _shading = _large_scale_fading+pathloss
        #_distributions = sorted(distributions, key=lambda name: stats.kstest(_shading, name))
        _distributions = distribution_fit(_shading)
        print(_distributions[:2])

        print((
               f'{w:>6} | '
               f'{np.std(_shading):13.2f} | '
               f'{np.mean(_shading):5.2f} | '
               f'{error(expected_shading, _shading):10.2f}'
        ))
