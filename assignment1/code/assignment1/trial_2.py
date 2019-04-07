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

    CGI, CGQ = fft(randn(Nfft)), fft(randn(Nfft))

    doppler_coeff = doppler_filter(fm,Nfft)

    f_CGI = CGI * np.sqrt(doppler_coeff)
    f_CGQ = CGQ * np.sqrt(doppler_coeff)

    tzeros = np.zeros(Nifft-Nfft)
    Filtered_CGI = np.hstack((f_CGI[:Nfft//2], tzeros, f_CGI[Nfft//2:]))
    Filtered_CGQ = np.hstack((f_CGQ[:Nfft//2], tzeros, f_CGQ[Nfft//2:]))

    hI, hQ = ifft(Filtered_CGI), ifft(Filtered_CGQ)

    rayEnvelope = np.abs(np.abs(hI) + 1j * hQ)
    rayRMS = math.sqrt(np.mean(rayEnvelope[:N]**2))

    # h_{I}(t) + jh_{Q}(t)
    h = (np.real(hI[:N]) - 1j * np.real(hQ[:N]))/rayRMS

    return h

if __name__ == '__main__':
    fm=100
    ts_mu=50
    scale=1e-6
    ts=ts_mu*scale
    fs=1/ts
    Nd=1e6

    h = fwgn_model(fm,fs,Nd)


    plt.subplot(211)
    plt.plot(np.arange(1,Nd+1)*ts,10*np.log10(np.abs(h)))
    plt.xlim(0, 0.5)
    plt.ylim(-30, 5)
    plt.title(f'Canal simulado com o modelo de Clarke/Gan')
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
    plt.savefig('gans.eps')
