import sympy as sp
import numpy as np
import pandas as pd
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

def real_signals(filenames):
    for filename in filenames:
        df = pd.read_csv(filename)
        
        cols = df.columns.tolist()
        if len(cols) >= 2:
            time = df[cols[0]].values
            signal = df[cols[1]].values
        else:
            signal = df[cols[0]].values
            dt = 1.0 / 5000.0
            time = np.arange(len(signal)) * dt
        
        dt = np.median(np.diff(time))
        Fs = 1 / dt
        N = len(signal)
        x = signal - np.mean(signal)
        
        print(f"Количество отсчетов: {N}")
        print(f"Частота дискретизации: {Fs:.2f} Гц")
        
        # === 3. FFT ===
        Nfft = 1 << (int(np.ceil(np.log2(N))))  # ближайшая степень двойки
        X = np.fft.rfft(x, n=Nfft)
        freqs = np.fft.rfftfreq(Nfft, d=dt)
        amplitude = (2.0 / N) * np.abs(X)
        phase = np.angle(X)

        # === 4. Поиск основной гармоники ===
        valid_idx = np.where(freqs > 1)[0]
        main_idx = valid_idx[np.argmax(amplitude[valid_idx])]
        f1 = freqs[main_idx]
        A1 = amplitude[main_idx]
        phi1 = phase[main_idx]
        phi1_deg = np.degrees(phi1)
        if phi1_deg > 180: phi1_deg -= 360

        # === 5. Временные параметры ===
        pk2pk = np.max(signal) - np.min(signal)
        A_time = pk2pk / 2
        tau = phi1 / (2 * np.pi * f1)

        # === 6. ВЫВОД РЕЗУЛЬТАТОВ ===
        print("\n--- РЕЗУЛЬТАТЫ ---")
        print(f"Основная частота: {f1:.3f} Гц")
        print(f"Амплитуда (по спектру): {A1:.3f}")
        print(f"Пиковая амплитуда во времени: {A_time:.3f}")
        print(f"Фаза: {phi1:.3f} рад ({phi1_deg:.2f}°)")
        print(f"Эквивалентная задержка: {tau*1000:.3f} мс")

        # === 7. ВИЗУАЛИЗАЦИЯ ===
        plt.figure(figsize=(10, 4))
        plt.plot(time, signal)
        plt.title("Сигнал во временной области")
        plt.xlabel("Время, с")
        plt.ylabel("Амплитуда")
        plt.grid(True)

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, amplitude)
        plt.title("Амплитудный спектр (односторонний)")
        plt.xlabel("Частота, Гц")
        plt.ylabel("Амплитуда")
        plt.xlim(0, Fs / 2)
        plt.grid(True)
        plt.axvline(f1, color='r', linestyle='--', label=f"f₁ ≈ {f1:.1f} Гц")
        plt.legend()

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, np.unwrap(phase))
        plt.title("Фазовый спектр")
        plt.xlabel("Частота, Гц")
        plt.ylabel("Фаза, рад")
        plt.xlim(0, Fs / 2)
        plt.grid(True)

        plt.show()

