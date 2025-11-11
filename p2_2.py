import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def test_signal(f0, fs, T, x, N):
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N, 1/fs)

    N2 = 200
    x_padded = np.pad(x, (0, N2 - N), 'constant')
    X_padded = np.fft.fft(x_padded)
    freq2 = np.fft.fftfreq(N2, 1/fs)

    window = np.hamming(N)
    x_win = x * window
    X_win = np.fft.fft(x_win, n = N2)
    freq_win = np.fft.fftfreq(N2, 1/fs)

    plt.figure()
    plt.subplot(2,1,1)
    plt.stem(freq, np.abs(X))
    plt.title("График с N=10 (Амплитуда)")

    plt.subplot(2,1,2)
    plt.stem(freq, np.angle(X))
    plt.title("График с N=10 (Фаза)")
    plt.savefig('magnitude_phase_N10.png', dpi=300)

    plt.figure()
    plt.subplot(2,1,1)
    plt.stem(freq2, np.abs(X_padded))
    plt.title("График с N=200 (Амплитуда)")

    plt.subplot(2,1,2)
    plt.stem(freq2, np.angle(X_padded))
    plt.title("График с N=200 (Фаза)")
    plt.savefig('magnitude_phase_N200.png', dpi=300)

    plt.figure()
    plt.subplot(2,1,1)
    plt.stem(freq_win, np.abs(X_win))
    plt.title("График с окном (Амплитуда)")

    plt.subplot(2,1,2)
    plt.stem(freq_win, np.angle(X_win))
    plt.title("График с окном (Фаза)")
    plt.savefig('magnitude_phase_window.png', dpi=300)

    plt.show()


