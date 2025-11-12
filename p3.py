import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows

def test_signals_magnitude_and_spectrogram(y1, y2, y3, fs, t, show_graphs):
    t_digit = np.arange(0, 3, 1/fs)

    y1 = sp.lambdify(t, y1, 'numpy')
    y2 = sp.lambdify(t, y2, 'numpy')
    y3 = sp.lambdify(t, y3, 'numpy')

    y1_d = y1(t_digit)
    y2_d = y2(t_digit)
    y3_d = y3(t_digit)

    N = len(t_digit)
    freqs = np.fft.fftfreq(N, 1/fs)
    Y1 = np.fft.fft(y1_d)
    Y2 = np.fft.fft(y2_d)
    Y3 = np.fft.fft(y3_d)

    amp1 = np.abs(Y1) * 2/N
    amp2 = np.abs(Y2) * 2/N
    amp3 = np.abs(Y3) * 2/N

    plt.figure()
    plt.stem(freqs, amp1)
    plt.title('Амплитудный спектр y_1(t)')
    plt.savefig("./p3/y1magnitude.png")

    plt.figure()
    plt.stem(freqs, amp2)
    plt.title('Амплитудный спектр y_2(t)')
    plt.savefig("./p3/y2magnitude.png")

    plt.figure()
    plt.stem(freqs, amp3)
    plt.title('Амплитудный спектр y_3(t)')
    plt.savefig("./p3/y3magnitude.png")

    plt.tight_layout()
    if show_graphs ==True:
        plt.show()
    else:
        plt.close()
    
    window_sizes = [int(0.01 * N), int(0.1 * N), int(0.3 * N)]
    window_sizes = [max(8, w) for w in window_sizes]  

    def plot_stft(y, title_prefix):
        plt.figure(figsize=(12, 8))
        for i, wlen in enumerate(window_sizes):
            win = windows.hamming(wlen)
            f, t_stft, Zxx = stft(y, fs=fs, window=win, nperseg=wlen, noverlap=wlen//2)
            plt.subplot(3, 1, i+1)
            plt.ylim(0, 10)
            plt.xlim(0, 3)
            plt.pcolormesh(t_stft, f, np.abs(Zxx), cmap='gnuplot', shading='auto')
            plt.title(f"{title_prefix} — окно Хемминга длиной {int(wlen/N*100)}% ({wlen} отсчётов)")
            plt.ylabel('Частота [Гц]')
            plt.xlabel('Время [с]')
            plt.colorbar(label='Амплитуда')
        plt.tight_layout()
        plt.savefig(f"./p3/{title_prefix}spectrogram_{int(wlen/N*100)}.png", dpi=300)
        if show_graphs ==True:
            plt.show()
        else:
            plt.close()

    plot_stft(y1_d, "y_1(t) — кусочная синусоида")
    plot_stft(y2_d, "y_2(t) — сумма трёх синусоид")
    plot_stft(y3_d, "y_3(t) — sin(2π(t+0.5)t)")