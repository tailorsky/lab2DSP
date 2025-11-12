import sympy as sp
import numpy as np

import p1
import p2
import p2_2
import p3

#ПУНКТ 1
t, w, t0, w0, k, a, b, tau, xi = sp.symbols('t w t0 w0 k a b tau xi', real=True)
I = sp.I
Abs = sp.Abs

f = sp.Function('f')(t)
g = sp.Function('g')(t)

F = sp.Function('F')(w)
G = sp.Function('G')(w)

p1.start(t, w, t0, w0, k, a, b, tau, xi, I, Abs, f, g, F, G)

#ПУНКТ 2 (НЕ ЗНАЮ, СТОИТ ЛИ НАСТОЛЬКО ПОДРОБНО ДЕЛАТЬ ЭТО ВСЁ)
Fs = 1000
Ts = 1/Fs
T = 1
t_numeric = np.arange(0, T, Ts)
N = len(t_numeric)

p2.start(Fs, Ts, T, N)
p2.prove_linearity(t_numeric, T, N, Ts)
p2.prove_convolution()
p2.prove_shift(t_numeric, N, Ts)
p2.prove_multiplication(t_numeric, N, Ts)
p2.prove_parseval(t_numeric, N)

#ПУНКТ 2_2
f0 = 1
fs = 10
T = 1
N = int (fs*T)
t = np.arange(N) / fs
x = np.exp(1j * 2 * np.pi * f0 * t)
p2_2.test_signal(f0, fs, T, x, N)
filenames = ["100hz.csv", "2.5khz.csv", "5khz.csv", "10khz.csv"]
p2_2.real_signals(filenames)

#ПУНКТ 3
f1, f2, f3 = 1,2,3 
sin = sp.sin
t = sp.Symbol('t', real=True)

y1 = sp.Piecewise(
    (sin(2 * sp.pi * f1 * t), (t >= 0) & (t < 1)),
    (sin(2 * sp.pi * f2 * t), (t >= 1) & (t < 2)),
    (sin(2 * sp.pi * f3 * t), (t >= 2) & (t < 3)),
)
y2 = 1/3*(sin(2*sp.pi*f1*t) + sin(2*sp.pi*f2*t) + sin(2*sp.pi*f3*t))
y3 = sin(2 * sp.pi*(t+0.5)*t)

fs = 1000
p3.test_signals_magnitude_and_spectrogram(y1, y2, y3, fs, t)
