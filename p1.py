import sympy as sp


def start(t, w, t0, w0, k, a, b, tau, xi, I, Abs, f, g, F, G, filename='fourier_results.txt'):
    log = []
    
    def write(line=""):
        log.append(line)
    
    def formal_FT(expr):
        expr = sp.simplify(expr)

        repl = {
            sp.FourierTransform(f, t, w): F,
            sp.FourierTransform(g, t, w): G,
            sp.FourierTransform(f.subs(t, t - t0), t, w): F * sp.exp(-I * w * t0),
            sp.FourierTransform(f * sp.exp(I * w0 * t), t, w): F.subs(w, w - w0),
            sp.FourierTransform(f.subs(t, k * t), t, w): F.subs(w, w / k) / Abs(k),
            sp.FourierTransform(sp.Derivative(f, t), t, w): I * w * F,
            sp.FourierTransform(sp.integrate(f.subs(t, tau) * g.subs(t, t - tau), (tau, -sp.oo, sp.oo)), t, w): F * G,
            sp.FourierTransform(f * g, t, w): (1 / (2 * sp.pi)) * sp.integrate(F.subs(w, xi) * G.subs(w, w - xi),
                                                                            (xi, -sp.oo, sp.oo))
        }

        expr = expr.replace(
            lambda e: isinstance(e, sp.FourierTransform) and e.args[2] != w,
            lambda e: F.subs(w, e.args[2])
        )

        return expr.xreplace(repl)

    def check_property(lhs, rhs, name):
        lhs_f = formal_FT(lhs)
        rhs_f = formal_FT(rhs)
        direct_check = sp.simplify(lhs_f - rhs_f) == 0

        try:
            lhs_ifft = sp.inverse_fourier_transform(lhs_f, w, t)
            rhs_ifft = sp.inverse_fourier_transform(rhs_f, w, t)
            inverse_check = sp.simplify(lhs_ifft - rhs_ifft) == 0
        except:
            inverse_check = "не вычислено"

        print(f"{name}:")
        print(f"FT = {lhs_f}")
        print(f"Ожидаемое преобразование: {rhs_f}")
        print(f"Сравнение преобразований: {str(direct_check).lower()}")
        print(f"Сравнение обратных преобразований: {str(inverse_check).lower()}\n")
        write(f"{name}:")
        write(f"FT = {lhs_f}")
        write(f"Ожидаемое преобразование: {rhs_f}")
        write(f"Сравнение преобразований: {str(direct_check).lower()}")
        write(f"Сравнение обратных преобразований: {str(inverse_check).lower()}\n")


    lhs1 = sp.FourierTransform(a * f + b * g, t, w)
    rhs1 = a * sp.FourierTransform(f, t, w) + b * sp.FourierTransform(g, t, w)
    check_property(lhs1, rhs1, "1) Линейность")

    lhs2 = sp.FourierTransform(f.subs(t, t - t0), t, w)
    rhs2 = sp.FourierTransform(f, t, w) * sp.exp(-I * w * t0)
    check_property(lhs2, rhs2, "2) Сдвиг")

    lhs3 = sp.FourierTransform(f * sp.exp(I * w0 * t), t, w)
    rhs3 = sp.FourierTransform(f, t, w - w0)
    check_property(lhs3, rhs3, "3) Модуляция")

    lhs4 = sp.FourierTransform(f.subs(t, k * t), t, w)
    rhs4 = (1 / Abs(k)) * sp.FourierTransform(f, t, w / k)
    check_property(lhs4, rhs4, "4) Масштабирование")

    conv_expr = sp.integrate(f.subs(t, tau) * g.subs(t, t - tau), (tau, -sp.oo, sp.oo))
    lhs5 = sp.FourierTransform(conv_expr, t, w)
    rhs5 = sp.FourierTransform(f, t, w) * sp.FourierTransform(g, t, w)
    check_property(lhs5, rhs5, "5) Свёртка")

    lhs6 = sp.FourierTransform(f * g, t, w)
    rhs6 = (1 / (2 * sp.pi)) * sp.integrate(F.subs(w, xi) * G.subs(w, w - xi), (xi, -sp.oo, sp.oo))
    check_property(lhs6, rhs6, "6) Произведение")

    lhs7 = sp.FourierTransform(sp.Derivative(f, t), t, w)
    rhs7 = I * w * sp.FourierTransform(f, t, w)
    check_property(lhs7, rhs7, "7) Производная")

    print("--- Формальные символьные проверки завершены ---")
    write("--- Формальные символьные проверки завершены ---")

    # сохраняем в файл
    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(log))
    write(f"\nРезультаты сохранены в файл: {filename}")
