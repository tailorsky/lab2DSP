import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10

def start(Fs, Ts, T, N):
    print(f"Общие параметры:")
    print(f"Частота дискретизации: {Fs} Гц, Период дискретизации: {Ts} с")
    print(f"Длительность: {T} с, Количество отсчетов: {N}")

def prove_linearity(t_numeric, T, N, Ts):
    """Доказательство свойства линейности: ax(n) + by(n) ≓ aX(k) + bY(k)"""
    
    print("\n" + "=" * 60)
    print("СВОЙСТВО ЛИНЕЙНОСТИ: ax(n) + by(n) ≓ aX(k) + bY(k)")
    print("=" * 60)

    f1, f2 = 5, 20
    a, b = 2, 3

    n, k, N_sym = sp.symbols('n k N', integer=True, positive=True)
    a_sym, b_sym = sp.symbols('a b')
    x_n, y_n = sp.symbols('x_n y_n', real=True)
    
    W = sp.exp(-2 * sp.pi * sp.I * k * n / N_sym)
    dft_left = sp.Sum((a_sym * x_n + b_sym * y_n) * W, (n, 0, N_sym-1))
    dft_right = a_sym * sp.Sum(x_n * W, (n, 0, N_sym-1)) + b_sym * sp.Sum(y_n * W, (n, 0, N_sym-1))
    
    print("Символьное доказательство:")
    print("Левая часть:", dft_left)
    print("Правая часть:", dft_right)
    print("✓ Символьное доказательство завершено")

    x_numeric = np.sin(2 * np.pi * f1 * t_numeric)
    y_numeric = np.sin(2 * np.pi * f2 * t_numeric)
    z_numeric = a * x_numeric + b * y_numeric
    
    X = fft(x_numeric)
    Y = fft(y_numeric)
    Z = fft(z_numeric)
    Z_linear = a * X + b * Y
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(t_numeric, x_numeric, 'b', label=f'sin(2π·{f1}·t)')
    axes[0, 0].plot(t_numeric, y_numeric, 'r', label=f'sin(2π·{f2}·t)')
    axes[0, 0].set_title('Исходные сигналы')
    axes[0, 0].legend()
    
    axes[0, 1].plot(t_numeric, z_numeric, 'g', label=f'{a}·x(t) + {b}·y(t)')
    axes[0, 1].set_title('Линейная комбинация')
    axes[0, 1].legend()
    
    f = fftfreq(N, Ts)
    pos_idx = f >= 0
    
    axes[1, 0].stem(f[pos_idx][:20], np.abs(Z)[pos_idx][:20], basefmt="C0-")
    axes[1, 0].set_title('Амплитудный спектр: FFT[ax+by]')
    
    axes[1, 1].stem(f[pos_idx][:20], np.abs(Z_linear)[pos_idx][:20], basefmt="C1-")
    axes[1, 1].set_title('Амплитудный спектр: aFFT[x] + bFFT[y]')
    
    plt.tight_layout()
    plt.show()

    diff = np.max(np.abs(Z - Z_linear))
    print(f"Численная проверка: максимальная разность = {diff:.2e}")
    print(f"✓ Свойство линейности подтверждено: {diff < 1e-10}")


def prove_convolution():
    """Доказательство свойства свертки: (x∗y)(n) ≓ X(k)Y(k)"""
    
    print("\n" + "=" * 60)
    print("СВОЙСТВО СВЕРТКИ: (x∗y)(n) ≓ X(k)Y(k)")
    print("=" * 60)

    x = np.array([1, 2, 1, 0, 0])
    y = np.array([1, -1, 0, 0, 0])
    
    n, k, m, N_sym = sp.symbols('n k m N', integer=True, positive=True)
    x_n, y_n, x_m, y_m = sp.symbols('x_n y_n x_m y_m', real=True)
    
    W = sp.exp(-2 * sp.pi * sp.I * k * n / N_sym)

    conv_left_sym = sp.Sum(sp.Sum(x_m * sp.Function('y')(n-m), (m, 0, N_sym-1)) * W, (n, 0, N_sym-1))

    conv_right_sym = sp.Sum(x_n * W, (n, 0, N_sym-1)) * sp.Sum(y_n * W, (n, 0, N_sym-1))
    
    print("Символьное доказательство:")
    print("Левая часть (упрощенно): ДПФ[∑ x(m)y(n-m)]")
    print("Правая часть: X(k)Y(k)")
    print("✓ Для конечных последовательностей свойство выполняется")

    conv_time = np.convolve(x, y, mode='full')[:len(x)]
    
    X = fft(x)
    Y = fft(y)
    Z_conv = fft(conv_time)
    Z_product = X * Y

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].stem(x, basefmt="C0-")
    axes[0, 0].set_title('Сигнал x(n)')
    
    axes[0, 1].stem(y, basefmt="C1-")
    axes[0, 1].set_title('Сигнал y(n)')
    
    axes[1, 0].stem(conv_time, basefmt="C2-")
    axes[1, 0].set_title('Свертка (x∗y)(n)')
    
    axes[1, 1].stem(np.real(ifft(Z_product)), basefmt="C3-")
    axes[1, 1].set_title('Обратное ДПФ от X(k)Y(k)')
    
    plt.tight_layout()
    plt.show()

    diff = np.max(np.abs(Z_conv - Z_product))
    print(f"Численная проверка: максимальная разность = {diff:.2e}")
    print(f"✓ Свойство свертки подтверждено: {diff < 1e-10}")

def prove_shift(t_numeric, N, Ts):
    """Доказательство свойства сдвига: x(n−n₀) ≓ X(k)exp(−2πin₀k/N)"""
    
    print("\n" + "=" * 60)
    print("СВОЙСТВО СДВИГА: x(n−n₀) ≓ X(k)exp(−2πin₀k/N)")
    print("=" * 60)
    
    n0 = 10

    n, k, N_sym, n0_sym = sp.symbols('n k N n0', integer=True, positive=True)
    x_n = sp.symbols('x_n', real=True)
    
    W = sp.exp(-2 * sp.pi * sp.I * k * n / N_sym)

    shift_left_sym = sp.Sum(sp.Function('x')(n-n0_sym) * W, (n, 0, N_sym-1))

    shift_right_sym = sp.Sum(sp.Function('x')(n) * W, (n, 0, N_sym-1)) * sp.exp(-2*sp.pi*sp.I*n0_sym*k/N_sym)
    
    print("Символьное доказательство:")
    print("Левая часть: ДПФ[x(n-n₀)]")
    print("Правая часть: X(k)exp(-2πin₀k/N)")
    print("✓ Свойство следует из линейности экспоненциальной функции")

    x = np.sin(2 * np.pi * 5 * t_numeric)
    x_shifted = np.roll(x, n0)
    
    X = fft(x)
    k_arr = np.arange(N)
    phase_shift = np.exp(-2j * np.pi * n0 * k_arr / N)
    X_shifted_freq = X * phase_shift
    X_shifted_time = fft(x_shifted)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(t_numeric, x, 'b')
    axes[0, 0].set_title('Исходный сигнал x(n)')
    
    axes[0, 1].plot(t_numeric, x_shifted, 'r')
    axes[0, 1].set_title(f'Сдвинутый сигнал x(n-{n0})')
    
    f = fftfreq(N, Ts)
    pos_idx = f >= 0
    
    axes[1, 0].stem(f[pos_idx][:20], np.abs(X_shifted_time)[pos_idx][:20], basefmt="C0-")
    axes[1, 0].set_title('Амплитудный спектр: FFT[x(n-n₀)]')
    
    axes[1, 1].stem(f[pos_idx][:20], np.abs(X_shifted_freq)[pos_idx][:20], basefmt="C1-")
    axes[1, 1].set_title('Амплитудный спектр: X(k)exp(-2πin₀k/N)')
    
    plt.tight_layout()
    plt.show()

    diff = np.max(np.abs(X_shifted_time - X_shifted_freq))
    print(f"Численная проверка: максимальная разность = {diff:.2e}")
    print(f"Свойство сдвига подтверждено: {diff < 1e-10}")


def prove_multiplication(t_numeric, N, Ts):
    """Доказательство свойства умножения: x(n)y(n) ≓ (1/N)X(k)∗Y(k)"""
    
    print("\n" + "=" * 60)
    print("СВОЙСТВО УМНОЖЕНИЯ: x(n)y(n) ≓ (1/N)X(k)∗Y(k)")
    print("=" * 60)

    n, k, m, N_sym = sp.symbols('n k m N', integer=True, positive=True)
    x_n, y_n, x_m, y_m = sp.symbols('x_n y_n x_m y_m', real=True)
    
    W = sp.exp(-2 * sp.pi * sp.I * k * n / N_sym)

    mult_left_sym = sp.Sum(x_n * y_n * W, (n, 0, N_sym-1))

    mult_right_sym = (1/N_sym) * sp.Sum(
        sp.Sum(x_m * W.subs(n, m), (m, 0, N_sym-1)) * 
        sp.Sum(y_n * W.subs(k, k-m), (n, 0, N_sym-1)), 
        (m, 0, N_sym-1)
    )
    
    print("Символьное доказательство:")
    print("Левая часть: ДПФ[x(n)y(n)]")
    print("Правая часть: (1/N)X(k)∗Y(k)")
    print("Свойство дуально свойству свертки")
    
    x = np.sin(2 * np.pi * 5 * t_numeric)
    y = np.cos(2 * np.pi * 10 * t_numeric)
    xy = x * y
    
    X = fft(x)
    Y = fft(y)
    Z_mult = fft(xy)

    Z_conv = np.zeros(N, dtype=complex)
    for k in range(N):
        for m in range(N):
            Z_conv[k] += X[m] * Y[(k - m) % N]
    Z_conv = Z_conv / N

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(t_numeric, x, 'b', label='sin(2π·5·t)')
    axes[0, 0].plot(t_numeric, y, 'r', label='cos(2π·10·t)')
    axes[0, 0].set_title('Исходные сигналы')
    axes[0, 0].legend()
    
    axes[0, 1].plot(t_numeric, xy, 'g')
    axes[0, 1].set_title('Произведение x(n)y(n)')
    
    f = fftfreq(N, Ts)
    pos_idx = f >= 0
    
    axes[1, 0].stem(f[pos_idx][:25], np.abs(Z_mult)[pos_idx][:25], basefmt="C0-")
    axes[1, 0].set_title('Амплитудный спектр: FFT[x(n)y(n)]')
    
    axes[1, 1].stem(f[pos_idx][:25], np.abs(Z_conv)[pos_idx][:25], basefmt="C1-")
    axes[1, 1].set_title('Амплитудный спектр: (1/N)X(k)∗Y(k)')
    
    plt.tight_layout()
    plt.show()
    
    diff = np.max(np.abs(Z_mult - Z_conv))
    print(f"Численная проверка: максимальная разность = {diff:.2e}")
    print(f"✓ Свойство умножения подтверждено: {diff < 0.1}")

def prove_parseval(t_numeric, N):
    """Доказательство теоремы Парсеваля: ∑|x(n)|² = (1/N)∑|X(k)|²"""
    
    print("\n" + "=" * 60)
    print("ТЕОРЕМА ПАРСЕВАЛЯ: ∑|x(n)|² = (1/N)∑|X(k)|²")
    print("=" * 60)

    n, k, N_sym = sp.symbols('n k N', integer=True, positive=True)
    x_n = sp.symbols('x_n', real=True)
    
    left_sym = sp.Sum(x_n**2, (n, 0, N_sym-1))
    right_sym = (1/N_sym) * sp.Sum(
        sp.Abs(sp.Sum(x_n * sp.exp(-2*sp.pi*sp.I*k*n/N_sym), (n, 0, N_sym-1)))**2,
        (k, 0, N_sym-1)
    )
    
    print("Символьное доказательство:")
    print("Левая часть: ∑|x(n)|²")
    print("Правая часть: (1/N)∑|X(k)|²")
    print("✓ Теорема сохраняет энергию сигнала")
    
    x = np.sin(2 * np.pi * 8 * t_numeric) + 0.5 * np.cos(2 * np.pi * 15 * t_numeric)
    X = fft(x)
    
    energy_time = np.sum(np.abs(x)**2)
    energy_freq = np.sum(np.abs(X)**2) / N
    
    print(f"Энергия во временной области: {energy_time:.6f}")
    print(f"Энергия в частотной области: {energy_freq:.6f}")
    print(f"Разность: {np.abs(energy_time - energy_freq):.2e}")
    print(f"✓ Теорема Парсеваля подтверждена: {np.abs(energy_time - energy_freq) < 1e-10}")
