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
                    'plot_window': 100,
                    'windows': (10, 50, 100, 150, 200),
                    'P_0': 0,
                    'd_0': 5,
                },
                {
                    'file_name': 'real_world_prx',
                    'plot_window': 5,
                    'windows': (2, 5, 10),
                    'P_0': 45,
                    'd_0': 5,
                },
            )

    for signal in signals:
        P_0, d_0 = signal['P_0'], signal['d_0']
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
        # FIXME: Needs to investigate why c is different of P_0 for the
        # real world signal
        # assert c == P_0

        n = -m/10
        pathloss = P_0 -m*logdistance

        print((
             'Informação da perda de percurso\n'
             '-------------------------------\n'
            f'Coeficiente de perda de percurso: {n}'
            ))
        if expected_pathloss is not None:
            print(f'Erro quadrático da perda de percurso: {error(-expected_pathloss, m*logdistance)}')
        print()

        large_scale_fading = movmean(prx, signal['plot_window'])
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


        plt.figure()
        title = 'Perda de percurso'
        if expected_pathloss is not None:
            title += ' original e estimada'
        plt.title(title)
        plt.plot(distance, prx, 'orange',
                 label='Potência recebida completa')
        if expected_pathloss is not None:
            plt.plot(distance, expected_pathloss, 'cyan',
                     label='Perda de percurso original')
        plt.plot(distance, -pathloss, 'r--',
                 label='Perda de percurso estimada')

        plt.xlabel('Distância (m)')
        plt.ylabel('Potência (dB)')
        plt.xlim(min(distance), max(distance))
        plt.legend()

        plt.savefig(f'trial1_{file_name}_pathloss.eps')


        plt.figure()
        title = 'Perda de percurso e sombreamento'
        if expected_pathloss is not None:
            title += ' originais e estimados'
        plt.title(title)
        plt.plot(distance, prx, 'orange',
                 label='Potência recebida completa')
        if expected_pathloss is not None:
            plt.plot(distance, expected_shading-expected_pathloss, 'cyan',
                     label='Perda de percurso e sombreamento originais')
        plt.plot(distance, shading-pathloss, 'r--',
                 label='Perda de percurso e sombreamento estimados')

        plt.xlabel('Distância (m)')
        plt.ylabel('Potência (dB)')
        plt.xlim(min(distance), max(distance))
        plt.legend()

        plt.savefig(f'trial1_{file_name}_pathloss_shading.eps')

        if expected_shading is not None:
            _plot_expected_shading(file_name, distance, prx,
                                   shading, expected_shading)
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

            _distribution = distribution_fit(_small_scale_fading)[0]

            distribution = getattr(stats, _distribution.name)
            params = distribution.fit(_small_scale_fading, *_distribution.args)

            print((
                   f'{w:>6} | '
                   f'{_distribution.name:>28} | '
                   f'{params[-2]:5.2f} | '
                   f'{params[-1]:6.2f} | '
                   f'{params[:-2]}'
            ))
        print(
               'Janela | '
               'Segunda melhor distribuição | Média | Escala | Argumentos adicionais'
            )
        for i, _large_scale_fading in enumerate(large_scale_fadings):
            w = windows[i]

            _small_scale_fading = prx - _large_scale_fading

            _distribution = distribution_fit(_small_scale_fading)[1]

            distribution = getattr(stats, _distribution.name)
            params = distribution.fit(_small_scale_fading, *_distribution.args)

            print((
                   f'{w:>6} | '
                   f'{_distribution.name:>27} | '
                   f'{params[-2]:5.2f} | '
                   f'{params[-1]:6.2f} | '
                   f'{params[:-2]}'
            ))
