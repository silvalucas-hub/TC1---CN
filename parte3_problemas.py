"""Parte 3 - Situações-problema (A-E) + bônus F do TC1."""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metodos import bisseccao, newton, secante

# Garante que a pasta usada para salvar os gráficos exista no Colab.
os.makedirs("figuras", exist_ok=True)


def intervalos_mudanca_sinal(f, xs):
    ys = [f(float(x)) for x in xs]
    ints = []
    for i in range(len(xs)-1):
        if ys[i] == 0:
            ints.append((float(xs[i]), float(xs[i])))
        elif ys[i+1] == 0:
            # A raiz exata será registrada na próxima posição da malha; evita intervalo duplicado.
            continue
        elif (ys[i] > 0) != (ys[i+1] > 0):
            ints.append((float(xs[i]), float(xs[i+1])))
    if ys[-1] == 0:
        ints.append((float(xs[-1]), float(xs[-1])))
    return ints


# -----------------------------------------------------------------------------
# Problema A - Reservatório esférico
# -----------------------------------------------------------------------------
def volume_esfera(h, R=3.0):
    return math.pi * h*h * (3*R - h) / 3.0


def problema_A(pasta_figuras="figuras"):
    print("\n=== Problema A - Reservatório esférico ===")
    R = 3.0
    alvo = 40.0
    F = lambda h: volume_esfera(h, R) - alvo
    dF = lambda h: math.pi * h * (2*R - h)

    # Fase I: faixa ampla para exibir as 3 raízes da cúbica.
    xs = np.linspace(-5, 12, 2001)
    ints = intervalos_mudanca_sinal(F, xs)
    print("Intervalos com mudança de sinal:", ints)

    raizes = []
    for a,b in ints:
        r, h = bisseccao(F, a, b, eps=1e-12)
        raizes.append((r, len(h)))
    print("Raízes reais:", raizes)

    # Raiz física: 0 <= h <= 2R
    fisicas = [r for r,_ in raizes if 0 <= r <= 2*R]
    h40 = fisicas[0]
    print(f"A.1 h físico = {h40:.12f} m; V(h)={volume_esfera(h40,R):.12f} m³")

    # A.3
    linhas = []
    for V in range(10, 111, 10):
        FV = lambda h, V=V: volume_esfera(h, R) - V
        r, hist = bisseccao(FV, 0.0 + 1e-12, 2*R - 1e-12, eps=1e-10)
        linhas.append({"V_m3": V, "h_m": r, "iteracoes": len(hist)})
    tab = pd.DataFrame(linhas)
    print(tab.to_string(index=False))

    plt.figure()
    plt.plot(tab["V_m3"], tab["h_m"], marker="o")
    plt.xlabel("Volume V (m³)")
    plt.ylabel("Altura h (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_figuras}/problema_A_h_vs_V.png", dpi=180)
    plt.close()
    return {"raizes": raizes, "h40": h40, "tabela": tab}


# -----------------------------------------------------------------------------
# Problema B - Colebrook-White
# -----------------------------------------------------------------------------
def problema_B(pasta_figuras="figuras"):
    print("\n=== Problema B - Perda de carga ===")
    D = 0.100
    rug = 4.5e-5
    Re = 2.0e5
    L = 500.0
    Q = 0.050
    g = 9.81
    Aconst = rug/(3.7*D)
    Bconst = 2.51/Re

    def F(ff):
        if ff <= 0:
            return float("nan")
        return 1/math.sqrt(ff) + 2*math.log10(Aconst + Bconst/math.sqrt(ff))

    def dF(ff):
        s = Aconst + Bconst/math.sqrt(ff)
        return -1/(2*ff**1.5) - Bconst/(math.log(10)*ff**1.5*s)

    xs = np.linspace(0.005, 0.08, 500)
    ints = intervalos_mudanca_sinal(F, xs)
    print("Fase I:", ints)
    bracket = ints[0]

    rb, hb = bisseccao(F, bracket[0], bracket[1], eps=1e-12)

    # Newton com o chute arbitrário f0=0,05 sugerido no enunciado sai do domínio
    # físico logo na primeira atualização. Registramos a falha e, para a comparação
    # de custo dos três métodos, usamos também um chute arbitrário viável f0=0,02.
    falha_newton_005 = None
    try:
        rn_005, hn_005 = newton(F, dF, 0.05, eps=1e-12)
    except Exception as exc:
        rn_005, hn_005 = None, None
        falha_newton_005 = str(exc)

    rn, hn = newton(F, dF, 0.02, eps=1e-12)
    rs, hs = secante(F, 0.05, 0.02, eps=1e-12)
    print("Bissecção", rb, len(hb), hb[-1]["chamadas_f"])
    print("Newton (x0=0,02)", rn, len(hn), hn[-1]["chamadas_f"], hn[-1]["chamadas_df"])
    print("Newton (x0=0,05):", "falhou - " + falha_newton_005 if falha_newton_005 else rn_005)
    print("Secante (0,05;0,02)", rs, len(hs), hs[-1]["chamadas_f"])

    f_sj = 0.25 / (math.log10(rug/(3.7*D) + 5.74/(Re**0.9))**2)
    rn_sj, hn_sj = newton(F, dF, f_sj, eps=1e-12)
    rs_sj, hs_sj = secante(F, f_sj*0.98, f_sj*1.02, eps=1e-12)
    print("Swamee-Jain f0=", f_sj)
    print("Newton com SJ:", len(hn_sj), "iterações; com 0,02:", len(hn), "; com 0,05: falha")
    print("Secante com SJ:", len(hs_sj), "vs arbitrário", len(hs))

    area = math.pi*D**2/4
    V = Q/area
    hf = rb*(L/D)*(V**2/(2*g))
    hf_2casas = 0.02*(L/D)*(V**2/(2*g))
    erro_pct = abs(hf_2casas-hf)/abs(hf)*100
    print(f"V={V:.8f} m/s, hf={hf:.8f} m, hf(f=0.02)={hf_2casas:.8f} m, erro={erro_pct:.4f}%")
    return {
        "f": rb, "hb": hb, "hn": hn, "hs": hs,
        "falha_newton_005": falha_newton_005,
        "f_sj": f_sj, "hn_sj": hn_sj, "hs_sj": hs_sj,
        "V": V, "hf": hf, "hf_2casas": hf_2casas, "erro_pct": erro_pct,
    }


# -----------------------------------------------------------------------------
# Problema C - van der Waals
# -----------------------------------------------------------------------------
def problema_C(pasta_figuras="figuras"):
    print("\n=== Problema C - van der Waals ===")
    R = 8.314
    T = 300.0
    a = 0.3640
    b = 4.267e-5
    P = 5.0e6

    v_ideal = R*T/P

    def F(v, P=P):
        return P*v**3 - (P*b + R*T)*v**2 + a*v - a*b

    xs = np.logspace(-5, -2, 5000)
    ints = intervalos_mudanca_sinal(F, xs)
    roots = []
    for lo,hi in ints:
        r,h = bisseccao(F, lo, hi, eps=1e-14, max_iter=300)
        roots.append((r,len(h)))
    print("v ideal =", v_ideal)
    print("intervalos =", ints)
    print("raízes =", roots)

    # Para comparação com gás ideal, quando houver várias raízes, usa a maior (ramo vapor).
    if roots:
        v_vdw = max(r for r,_ in roots)
    else:
        raise RuntimeError("Nenhuma raiz positiva encontrada.")
    erro_ideal = abs(v_ideal-v_vdw)/abs(v_vdw)*100
    print(f"v_vdW(gas-like)={v_vdw:.12e}; erro ideal={erro_ideal:.6f}%")

    # C.4: isoterma P x v; maior raiz positiva quando houver mais de uma.
    linhas = []
    for P_MPa in np.arange(1.0, 10.0+0.001, 0.5):
        P_i = P_MPa*1e6
        def Fi(v, P_i=P_i):
            return P_i*v**3 - (P_i*b + R*T)*v**2 + a*v - a*b
        xs_i = np.logspace(math.log10(b*1.0001), -2, 4000)
        ints_i = intervalos_mudanca_sinal(Fi, xs_i)
        roots_i = []
        for lo,hi in ints_i:
            r,_ = bisseccao(Fi, lo, hi, eps=1e-13, max_iter=300)
            if r > b:
                roots_i.append(r)
        if not roots_i:
            v_vdw_i = float("nan")
        else:
            v_vdw_i = max(roots_i)
        v_id_i = R*T/P_i
        linhas.append({"P_MPa": P_MPa, "v_vdw": v_vdw_i, "v_ideal": v_id_i, "n_raizes": len(roots_i)})
    tab = pd.DataFrame(linhas)
    print(tab.to_string(index=False))

    plt.figure()
    plt.plot(tab["v_vdw"], tab["P_MPa"], marker="o", label="van der Waals (maior raiz)")
    plt.plot(tab["v_ideal"], tab["P_MPa"], marker="s", label="gás ideal")
    plt.xlabel("Volume molar v (m³/mol)")
    plt.ylabel("Pressão P (MPa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_figuras}/problema_C_isoterma.png", dpi=180)
    plt.close()
    return {"v_ideal": v_ideal, "roots": roots, "v_vdw": v_vdw, "erro_ideal": erro_ideal, "tabela": tab}


# -----------------------------------------------------------------------------
# Problema D - TIR
# -----------------------------------------------------------------------------
def problema_D(pasta_figuras="figuras"):
    print("\n=== Problema D - TIR ===")
    fluxos = [-1000, 300, 350, 400, 450]

    def vpl(i, fluxos=fluxos):
        if i <= -1:
            return float("nan")
        return sum(C/((1+i)**k) for k,C in enumerate(fluxos))

    xs = np.linspace(0,0.5,501)
    ints = intervalos_mudanca_sinal(vpl, xs)
    print("Fase I projeto 1:", ints)
    tir, hist = bisseccao(vpl, ints[0][0], ints[0][1], eps=1e-10)
    print("TIR =", tir, "iterações", len(hist))
    vpl15 = vpl(0.15)
    vpl20 = vpl(0.20)
    print("VPL 15% =", vpl15, "VPL 20% =", vpl20)

    xx = np.linspace(0,0.5,400)
    yy = [vpl(x) for x in xx]
    plt.figure()
    plt.plot(xx, yy)
    plt.axhline(0, linewidth=1)
    plt.xlabel("taxa i")
    plt.ylabel("VPL (mil R$)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_figuras}/problema_D_vpl1.png", dpi=180)
    plt.close()

    fluxos2 = [-1000,2500,-1540]
    def vpl2(i):
        if i <= -1:
            return float("nan")
        return sum(C/((1+i)**k) for k,C in enumerate(fluxos2))
    xx2 = np.linspace(0,0.6,601)
    ints2 = intervalos_mudanca_sinal(vpl2, xx2)
    roots2 = []
    for lo,hi in ints2:
        if lo == hi:
            roots2.append((lo,0))
        else:
            r,h = bisseccao(vpl2,lo,hi,eps=1e-12)
            roots2.append((r,len(h)))
    print("Projeto 2 intervalos:", ints2, "raízes:", roots2)

    yy2 = [vpl2(x) for x in xx2]
    plt.figure()
    plt.plot(xx2, yy2)
    plt.axhline(0, linewidth=1)
    plt.xlabel("taxa i")
    plt.ylabel("VPL (mil R$)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_figuras}/problema_D_vpl2.png", dpi=180)
    plt.close()
    return {"tir":tir,"hist":hist,"vpl15":vpl15,"vpl20":vpl20,"roots2":roots2}


# -----------------------------------------------------------------------------
# Problema E - Kepler
# -----------------------------------------------------------------------------
def resolver_kepler(e, M, x0, eps=1e-12):
    F = lambda E: E - e*math.sin(E) - M
    dF = lambda E: 1 - e*math.cos(E)
    return newton(F,dF,x0,eps=eps,max_iter=200)


def problema_E():
    print("\n=== Problema E - Kepler ===")
    # E.1
    r1,h1 = resolver_kepler(0.967,0.2,0.2)
    print("E.1:",r1,len(h1),"resíduo", r1-0.967*math.sin(r1)-0.2)

    casos = [(0.10,0.5),(0.90,0.1),(0.99,0.01)]
    linhas=[]
    resultados=[]
    for e,M in casos:
        try:
            r,h=resolver_kepler(e,M,M)
            linhas.append({"e":e,"M":M,"E":r,"iteracoes":len(h),"convergiu":h[-1]["convergiu"]})
            resultados.append((r,h))
        except Exception as exc:
            linhas.append({"e":e,"M":M,"E":float('nan'),"iteracoes":0,"convergiu":False})
            resultados.append((None,exc))
    tab=pd.DataFrame(linhas)
    print(tab.to_string(index=False))

    e,M=0.99,0.01
    chute_bom=M+e*math.sin(M)
    r3,h3=resolver_kepler(e,M,chute_bom)
    print("E.3 chute melhor",chute_bom,"->",r3,"iter",len(h3))

    F=lambda E:E-e*math.sin(E)-M
    rb,hb=bisseccao(F,0,math.pi,eps=1e-12,max_iter=300)
    print("E.4 bissecção",rb,"iter",len(hb),"avaliações f",hb[-1]["chamadas_f"])
    return {"E1":(r1,h1),"tabela":tab,"chute_bom":chute_bom,"E3":(r3,h3),"E4":(rb,hb)}


# -----------------------------------------------------------------------------
# Problema F - BÔNUS: viga
# -----------------------------------------------------------------------------
def problema_F(pasta_figuras="figuras"):
    print("\n=== Problema F - Viga (bônus) ===")
    L=600.0
    E=50000.0
    I=30000.0
    w0=2.5
    C=w0/(120*L*E*I)
    def y(x):
        return C*(-x**5 + 2*L**2*x**3 - L**4*x)
    def dy(x):
        return C*(-5*x**4 + 6*L**2*x**2 - L**4)

    x_exato=L/math.sqrt(5)
    y_exato=y(x_exato)
    print("x interior exato=",x_exato,"cm; y=",y_exato,"cm; |y|=",abs(y_exato))
    print("dy(0)=",dy(0),"dy(L)=",dy(L))
    try:
        bisseccao(dy,0,L,eps=1e-10)
    except ValueError as e:
        print("F.3 [0,L]:",e)
    rb,hb=bisseccao(dy,0,L/2,eps=1e-10)
    print("intervalo corrigido [0,L/2]:",rb,"iter",len(hb))

    linhas=[]
    for h in [1e-2,1e-4,1e-6,1e-8,1e-10]:
        def dy_num(x,h=h):
            return (y(x+h)-y(x-h))/(2*h)
        try:
            r,hist=bisseccao(dy_num,0,L/2,eps=1e-9,max_iter=300)
            erro=abs(r-x_exato)
            linhas.append({"h":h,"x":r,"erro_cm":erro,"iteracoes":len(hist)})
        except Exception:
            linhas.append({"h":h,"x":float('nan'),"erro_cm":float('nan'),"iteracoes":0})
    tab=pd.DataFrame(linhas)
    print(tab.to_string(index=False))

    plt.figure()
    plt.loglog(tab["h"],tab["erro_cm"],marker="o")
    plt.xlabel("passo h")
    plt.ylabel("erro em x (cm)")
    plt.grid(True,which="both",alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_figuras}/problema_F_erro_derivada.png",dpi=180)
    plt.close()
    return {"x_exato":x_exato,"y_exato":y_exato,"tab":tab,"hb":hb}


def main():
    A=problema_A()
    B=problema_B()
    C=problema_C()
    D=problema_D()
    E=problema_E()
    F=problema_F()
    return A,B,C,D,E,F


if __name__ == "__main__":
    main()
