"""Parte 2 - Exercícios didáticos do TC1 de Cálculo Numérico."""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metodos import bisseccao, newton, secante


def f_padrao(x):
    return x**3 - 9*x + 3


def df_padrao(x):
    return 3*x**2 - 9


def tabelar_sinais(f, a, b, n):
    """Avalia f em n pontos igualmente espaçados e retorna intervalos com troca de sinal."""
    xs = np.linspace(a, b, n)
    valores = [f(x) for x in xs]
    intervalos = []
    for i in range(n - 1):
        fi, fj = valores[i], valores[i + 1]
        if fi == 0:
            intervalos.append((xs[i], xs[i]))
        elif (fi > 0) != (fj > 0):
            intervalos.append((xs[i], xs[i + 1]))
    if valores[-1] == 0:
        intervalos.append((xs[-1], xs[-1]))
    return intervalos


def exercicio_21():
    print("\n=== 2.1 - Isolamento ===")
    for n in [21, 11, 6, 4]:
        ints = tabelar_sinais(f_padrao, -5, 5, n)
        print(f"f, n={n}: {len(ints)} intervalo(s): {ints}")

    def g(x):
        return (x - 1.05) * (x - 1.15) * (x - 3)

    for n in [9, 17, 41, 401]:
        ints = tabelar_sinais(g, 0, 4, n)
        print(f"g, n={n}: {len(ints)} intervalo(s): {ints}")


def exercicio_22():
    print("\n=== 2.2 - Previsão x realidade na bissecção ===")
    linhas = []
    for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        valor = math.log((1 - 0) / eps, 2)
        # O enunciado usa k > valor: menor inteiro estritamente maior.
        previsto = math.floor(valor) + 1
        raiz, hist = bisseccao(f_padrao, 0, 1, eps=eps)
        linhas.append({"eps": eps, "previsto": previsto, "efetivo": len(hist), "raiz": raiz})
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))
    return tabela


def exercicio_23():
    print("\n=== 2.3 - Custo real ===")
    eps = 1e-8
    rb, hb = bisseccao(f_padrao, 0, 1, eps=eps)
    rn, hn = newton(f_padrao, df_padrao, 0.5, eps=eps)
    rs, hs = secante(f_padrao, 0, 1, eps=eps)
    linhas = []
    for nome, raiz, hist in [
        ("Bissecção", rb, hb), ("Newton", rn, hn), ("Secante", rs, hs)
    ]:
        ult = hist[-1]
        linhas.append({
            "Método": nome,
            "Raiz": raiz,
            "Iterações": len(hist),
            "Avaliações f": ult["chamadas_f"],
            "Avaliações f'": ult.get("chamadas_df", 0),
        })
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))
    return tabela


def ordens_empiricas(hist, raiz_exata):
    erros = [abs(item["x"] - raiz_exata) for item in hist]
    ordens = [float("nan")] * len(erros)
    for k in range(1, len(erros) - 1):
        if erros[k - 1] > 0 and erros[k] > 0 and erros[k + 1] > 0:
            den = math.log(erros[k] / erros[k - 1])
            if abs(den) > 1e-15:
                ordens[k] = math.log(erros[k + 1] / erros[k]) / den
    return erros, ordens


def exercicio_24(pasta_figuras="figuras"):
    os.makedirs(pasta_figuras, exist_ok=True)
    print("\n=== 2.4 - Ordem empírica de convergência ===")
    xi = 0.3376089559658377
    _, hn = newton(f_padrao, df_padrao, 0.5, eps=1e-14, max_iter=50)
    _, hs = secante(f_padrao, 0, 1, eps=1e-14, max_iter=50)
    en, pn = ordens_empiricas(hn, xi)
    es, ps = ordens_empiricas(hs, xi)

    tn = pd.DataFrame({"k": range(len(en)), "erro": en, "p_estimado": pn})
    ts = pd.DataFrame({"k": range(len(es)), "erro": es, "p_estimado": ps})
    print("Newton:\n", tn.to_string(index=False))
    print("Secante:\n", ts.to_string(index=False))

    plt.figure()
    plt.semilogy(range(len(en)), en, marker="o", label="Newton")
    plt.semilogy(range(len(es)), es, marker="s", label="Secante")
    plt.xlabel("iteração k")
    plt.ylabel(r"erro $|x_k-\xi|$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_figuras}/parte2_ordem_convergencia.png", dpi=180)
    plt.close()
    return tn, ts


def newton_iteracoes_fixas(f, df, x0, n=10):
    xs = [float(x0)]
    for _ in range(n):
        x = xs[-1]
        d = df(x)
        if abs(d) <= 1e-15:
            raise ZeroDivisionError(f"derivada nula em x={x}")
        xs.append(x - f(x) / d)
    return xs


def converge_arctan(x0, max_iter=100, limite=1e6):
    x = float(x0)
    for _ in range(max_iter):
        fx = math.atan(x)
        if abs(fx) < 1e-10:
            return True, x
        dfx = 1.0 / (1.0 + x*x)
        x = x - fx / dfx
        if not math.isfinite(x) or abs(x) > limite:
            return False, x
    return abs(math.atan(x)) < 1e-8, x


def limite_convergencia_arctan():
    # Busca a fronteira positiva entre os chutes que convergem e os que escapam.
    lo, hi = 0.0, 2.0
    # Encontrar intervalo transicional razoável.
    for _ in range(70):
        mid = (lo + hi) / 2
        ok, _ = converge_arctan(mid)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo, hi


def exercicio_25():
    print("\n=== 2.5 - Modos de falha de Newton ===")
    # (a)
    fa = lambda x: x**3 - 2*x + 2
    dfa = lambda x: 3*x**2 - 2
    xs = newton_iteracoes_fixas(fa, dfa, 0, 10)
    print("(a) x_k:", xs)

    # (b)
    fb = math.atan
    dfb = lambda x: 1/(1+x*x)
    for x0 in [2.0, 1.0]:
        try:
            r, h = newton(fb, dfb, x0, eps=1e-10, max_iter=30)
            print(f"(b) x0={x0}: raiz={r}, iterações={len(h)}, convergiu={h[-1]['convergiu']}")
        except Exception as e:
            print(f"(b) x0={x0}: falhou: {e}")
    lo, hi = limite_convergencia_arctan()
    print(f"(b) fronteira numérica aproximada de x0>0: entre {lo:.12f} e {hi:.12f}")

    # (c)
    try:
        r, h = newton(f_padrao, df_padrao, math.sqrt(3), eps=1e-8)
        print(f"(c) x0=sqrt(3): raiz={r}, iterações={len(h)}, convergiu={h[-1]['convergiu']}")
        print(f"    primeira iteração: x={h[0]['x']}, f(x)={h[0]['fx']}, erro={h[0]['erro']}")
    except (ZeroDivisionError, ValueError) as e:
        print("(c)", e)


def exercicio_26():
    print("\n=== 2.6 - Raiz múltipla ===")
    f = lambda x: (x - 2)**2 * (x + 1)
    df = lambda x: 2*(x-2)*(x+1) + (x-2)**2
    x = 3.0
    linhas = []
    for k in range(10):
        erro = abs(x - 2)
        x_novo = x - f(x)/df(x)
        erro_novo = abs(x_novo - 2)
        linhas.append({"k": k, "x": x, "erro": erro, "e_{k+1}/e_k": erro_novo/erro})
        x = x_novo
    tabela_normal = pd.DataFrame(linhas)
    print("Newton normal:\n", tabela_normal.to_string(index=False))

    x = 3.0
    linhas = []
    m = 2
    for k in range(10):
        erro = abs(x - 2)
        d = df(x)
        if abs(d) < 1e-15 or erro == 0:
            break
        x_novo = x - m*f(x)/d
        erro_novo = abs(x_novo - 2)
        linhas.append({"k": k, "x": x, "erro": erro, "erro_novo": erro_novo})
        x = x_novo
    tabela_mod = pd.DataFrame(linhas)
    print("Newton modificado:\n", tabela_mod.to_string(index=False))
    return tabela_normal, tabela_mod


def bisseccao_didatica_sem_validacao(f, a, b, eps=1e-8, criterio="residuo", max_iter=200):
    """Variante APENAS para 2.7: desliga a validação para expor a armadilha do exercício."""
    fa = f(a)
    x_ant = None
    hist = []
    for k in range(max_iter):
        c = (a+b)/2
        fc = f(c)
        erro_passo = float("inf") if x_ant is None else abs(c-x_ant)
        hist.append((k, c, fc, erro_passo))
        if criterio == "residuo" and abs(fc) < eps:
            return c, hist
        if criterio == "passo" and x_ant is not None and erro_passo < eps:
            return c, hist
        # Repete a lógica de sinal mesmo sem bracket válido, propositalmente.
        if (fa > 0) != (fc > 0):
            b = c
        else:
            a = c
            fa = fc
        x_ant = c
    return c, hist


def exercicio_27():
    print("\n=== 2.7 - Armadilha do resíduo ===")
    f = lambda x: (x - 1)**10
    print("f(1.1)=", f(1.1))
    print("f(1.3)=", f(1.3))
    try:
        bisseccao(f, 0, 1.5, eps=1e-8)
    except ValueError as e:
        print("Implementação oficial:", e)

    xr, hr = bisseccao_didatica_sem_validacao(f, 0, 1.5, 1e-8, "residuo")
    xp, hp = bisseccao_didatica_sem_validacao(f, 0, 1.5, 1e-8, "passo")
    print(f"Variante didática - resíduo: x={xr}, erro real={abs(xr-1)}, it={len(hr)}")
    print(f"Variante didática - passo:   x={xp}, erro real={abs(xp-1)}, it={len(hp)}")
    return xr, hr, xp, hp


def main():
    exercicio_21()
    exercicio_22()
    exercicio_23()
    exercicio_24()
    exercicio_25()
    exercicio_26()
    exercicio_27()


if __name__ == "__main__":
    main()
