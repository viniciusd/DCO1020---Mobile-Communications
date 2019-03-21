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
    return (np.linalg.norm(x-y)/np.linalg.norm(x))**2

def move_mean_to_0(signal, x):
    return signal-np.mean(signal)+np.mean(signal)*np.sin(2*np.pi*x)

def _plot_expected_prx(file_name, distance, prx, expected_pathloss, expected_shading):
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

    plt.savefig(f'trial1_{file_name}_original.eps')

def _plot_expected_pathloss(file_name, distance, prx, pathloss, expected_pathloss):
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

    plt.savefig(f'trial1_{file_name}_pathloss.eps')

def _plot_expected_shading(file_name, distance, prx, shading, expected_shading):
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

    plt.savefig(f'trial1_{file_name}_shading.eps')


if __name__ == '__main__':

    signals = (
                {
                    'file_name': 'prx',
                    'windows': (10, 50, 100, 150, 200),
                },
                {
                    'file_name': 'real_world_prx',
                    'windows': (2, 5, 10),
                },
            )

    P_0, d_0 = 0, 5
    for signal in signals:
        file_name = signal['file_name']
        signal_data = _load_signal(file_name)
        distance = signal_data['dPath']
        logdistance = np.log10(signal_data['dPath']/d_0)
        prx = signal_data.get('Prx', signal_data.get('dPrx'))
        expected_pathloss = signal_data.get('pathLoss')
        expected_shading = signal_data.get('shadCorr')


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
            f'Pathloss coeficient: {n}'
            ))
        if expected_pathloss is not None:
            print(f'Pathloss quadratic error: {error(-expected_pathloss, m*logdistance)}')
        print()

        large_scale_fading = movmean(prx, 100)
        small_scale_fading = prx - large_scale_fading
        shading = move_mean_to_0(large_scale_fading+pathloss, distance)


        if expected_pathloss is not None and expected_shading is not None:
            _plot_expected_prx(file_name, distance, prx,
                               expected_pathloss, expected_shading)

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

        plt.savefig(f'trial1_{file_name}_estimated.eps')


        if expected_pathloss is not None:
            _plot_expected_pathloss(file_name, distance, prx,
                                    pathloss, expected_pathloss)

        if expected_shading is not None:
            _plot_expected_shading(file_name, distance, prx,
                                   shading, expected_shading)
        """plt.figure()
        x = np.linspace(-30,30,1000)
        param = stats.dgamma.fit(shading, 5)
        pdf_fitted = stats.dgamma.pdf(x, param[0], loc=param[-2], scale=param[-1])
        plt.hist(shading, normed=True, label='Histograma do sombreamento')
        plt.plot(x,pdf_fitted,'r-', label='Fit da distribuição gamma')
        param = stats.cauchy.fit(shading)
        pdf_fitted = stats.cauchy.pdf(x, loc=param[-2], scale=param[-1])
        plt.plot(x,pdf_fitted,'b-', label='Fit da distribuição de Cauchy')
        plt.legend()
        plt.show()

        breakpoint()
        """
        print('\nSombreamento')
        print(
                'Janela | Desvio padrão |  Média | Erro Médio'
            )
        windows = signal['windows']
        large_scale_fadings = [movmean(prx,w) for w in windows]
        for i, _large_scale_fading in enumerate(large_scale_fadings):
            w = windows[i]

            _shading = _large_scale_fading+pathloss
            
            _error = error(expected_shading, _shading) if expected_shading is not None else np.nan

            print((
                   f'{w:>6} | '
                   f'{np.std(_shading):13.2f} | '
                   f'{np.mean(_shading):6.2f} | '
                   f'{_error:10.2f}'
            ))

        print('\nAderência estatística')
        print(
               'Janela | '
               'Primeira melhor distribuição | Média | Escala | Argumentos adicionais'
            )
        for i, _large_scale_fading in enumerate(large_scale_fadings):
            w = windows[i]

            _small_scale_fading = prx - _large_scale_fading

            first_name, second_name = '', ''
            if w == 10:
                first_name, second_name = 'genlogistic', 'dgamma'
                first_args, second_args = (1,), (2,)
            elif w == 50:
                first_name, second_name = 'dgamma', 'genlogistic'
                first_args, second_args = (2,), (1,)
            elif w == 100:
                first_name, second_name = 'dgamma', 'genlogistic'
                first_args, second_args = (2,), (1,)
            elif w == 150:
                first_name, second_name = 'dgamma', 'cauchy'
                first_args, second_args = (2,), ()
            elif w == 200:
                first_name, second_name = 'dgamma', 'cauchy'
                first_args, second_args = (3,), ()
            else:
                _distributions = distribution_fit(_small_scale_fading)
                first_name, second_name = _distributions[0].name, _distributions[1].name
                first_args, second_args = _distributions[0].args, _distributions[1].args

            first, second = getattr(stats, first_name), getattr(stats, second_name)
            first_params = first.fit(_small_scale_fading, *first_args)
            second_params = second.fit(_small_scale_fading, *second_args)

            print((
                   f'{w:>6} | '
                   f'{first_name:>28} | '
                   f'{first_params[-2]:5.2f} | '
                   f'{first_params[-1]:6.2f} | '
                   f'{first_params[:-2]}'
            ))
        print(
               'Janela | '
               'Segunda melhor distribuição | Média | Escala | Argumentos adicionais'
            )
        for i, _large_scale_fading in enumerate(large_scale_fadings):
            w = windows[i]

            _small_scale_fading = prx - _large_scale_fading

            first_name, second_name = '', ''
            if w == 10:
                first_name, second_name = 'genlogistic', 'dgamma'
                first_args, second_args = (1,), (2,)
            elif w == 50:
                first_name, second_name = 'dgamma', 'genlogistic'
                first_args, second_args = (2,), (1,)
            elif w == 100:
                first_name, second_name = 'dgamma', 'genlogistic'
                first_args, second_args = (2,), (1,)
            elif w == 150:
                first_name, second_name = 'dgamma', 'cauchy'
                first_args, second_args = (2,), ()
            elif w == 200:
                first_name, second_name = 'dgamma', 'cauchy'
                first_args, second_args = (3,), ()
            else:
                _distributions = distribution_fit(_small_scale_fading)
                first_name, second_name = _distributions[0].name, _distributions[1].name
                first_args, second_args = _distributions[0].args, _distributions[1].args

            first, second = getattr(stats, first_name), getattr(stats, second_name)
            first_params = first.fit(_small_scale_fading, *first_args)
            second_params = second.fit(_small_scale_fading, *second_args)

            print((
                   f'{w:>6} | '
                   f'{second_name:>27} | '
                   f'{second_params[-2]:5.2f} | '
                   f'{second_params[-1]:6.2f} | '
                   f'{second_params[:-2]}'
            ))
