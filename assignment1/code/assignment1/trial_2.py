import gc
import math
import matplotlib.pylab as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.stats import norm


def randn(n):
    return norm.ppf(np.random.rand(n))

def nextpow2(n):
    return math.ceil(np.log2(n))

def doppler_filter(fd,Nfft):
    f, y = np.zeros(Nfft//2), np.zeros(Nfft)
    df = 2*fd/Nfft

    for i in range(Nfft//2):
       f[i] = i*df
       y[i] = 1.5/(np.pi*fd*np.sqrt(1 - (f[i]/fd)**2))

    # It is symmetrical around Nfft//2
    y[Nfft//2+1:] = y[Nfft//2:1:-1]
    # We can now estimate the value in the middle based on a
    # third degree polynomial
    nFitPoints = 3
    kk = slice(Nfft//2-nFitPoints-1, Nfft//2)
    polyFreq = np.polyfit(f[kk], y[kk], nFitPoints)
    y[Nfft//2] = np.poly1d(polyFreq)(f[-1]+df)

    return y


def fwgn_model(fm,fs,N):
    N = int(N)
    Nfft = 2**max(3,nextpow2(2*fm/fs*N))
    Nifft = math.ceil(Nfft*fs/(2*fm))

    doppler_coeff = np.sqrt(doppler_filter(fm,Nfft))

    CGI, CGQ = fft(randn(Nfft)), fft(randn(Nfft))
    f_CGI = CGI * doppler_coeff
    f_CGQ = CGQ * doppler_coeff

    del CGI, CGQ, doppler_coeff
    gc.collect()

    tzeros = np.zeros(abs(Nifft-Nfft))
    filtered_CGI = np.hstack((f_CGI[:Nfft//2], tzeros, f_CGI[Nfft//2:]))
    filtered_CGQ = np.hstack((f_CGQ[:Nfft//2], tzeros, f_CGQ[Nfft//2:]))

    del tzeros, f_CGI, f_CGQ
    gc.collect()

    hI, hQ = ifft(filtered_CGI), ifft(filtered_CGQ)

    del filtered_CGI, filtered_CGQ
    gc.collect()

    rayEnvelope = np.abs(np.abs(hI) + 1j * hQ)
    rayRMS = math.sqrt(np.mean(rayEnvelope[:N]**2))

    # h_{I}(t) - jh_{Q}(t)
    # Here we have the phase shift of pi/2 when multiplying the imaginary
    # portion by -1j
    h = (np.real(hI[:N]) - 1j * np.real(hQ[:N]))/rayRMS

    return h

if __name__ == '__main__':
    for v in (3, 60, 120):
        v /= 3.6 # km/h -> m/s
        fm = 1900e6
        c = 3e8
        fd = v/(c/fm)
        fl = fm-fd
        fu = fm+fd
        n = 50
        fs = fl/(n-1)+fu/n
        ts = 1/fs
        Nd = 1e6

        h = fwgn_model(fm,fs,Nd)

        plt.subplot(211)
        time = np.arange(1,Nd+1)*ts
        plt.plot(np.arange(1,Nd+1)*ts,10*np.log10(np.abs(h)))
        plt.xlim(time.min(), time.max())
        plt.ylim(-30, 5)
        plt.title(f'Canal simulado com o modelo de Clarke/Gans')
        plt.xlabel('Tempo[s]')
        plt.ylabel('Magnitude[dB]')

        plt.subplot(223)
        magnitude = np.abs(h)
        plt.hist(magnitude, bins=50, density=True)
        plt.xlim(magnitude.min(), magnitude.max())
        plt.xlabel('Magnitude')
        plt.ylabel('Ocorrências')

        plt.subplot(224)
        angle = np.angle(h)
        plt.hist(angle, bins=50, density=True)
        plt.xlim(angle.min(), angle.max())
        plt.xlabel('Fase[rad]')
        plt.ylabel('Ocorrências')

        plt.tight_layout()
        plt.savefig(f'gans_{v}.eps')
