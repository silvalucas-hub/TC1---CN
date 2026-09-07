def contador(f):
    """ Retorna uma função que conta o número de chamadas a f. """
    def wrapper(x):
        wrapper.n += 1
        return f(x)
    wrapper.n = 0
    return wrapper

def bisseccao (f , a , b , eps =1e-8 , max_iter=200) :
    """ Retorna (raiz , historico ). """
    # Cria uma versão contadora da função f
    f_contada = contador(f)
    f = f_contada

    historico = []
    
    fa = f(a)
    fb = f(b)

    # Valida a entrada usando a sugestão do anexo para evitar underflow
    if (fa > 0) == (fb > 0):
        raise ValueError ("f(a) e f(b) devem ter sinais opostos.")

    c_ant = a # Valor inicial para o erro

    for i in range (max_iter) :
        c = (a + b) / 2.0
        fx = f(c)
        erro = abs(c - c_ant)
        atual = {
            "k": i,
            "x": c,
            "fx": fx,
            "erro": erro,
            "chamadas_f": f_contada.n, # Número de chamadas a f direto do decorator
            "convergiu": False
        }
        historico.append(atual)

        # Critério de parada
        if abs(fx) < eps or erro < eps :
            historico[-1]["convergiu"] = True
            return c , historico

        # Atualiza os limites do intervalo [a, b] para a próxima iteração
        if (fa > 0) != (fx > 0) :
            b = c
        else :
            a = c
            fa = fx

        c_ant = c # Atualiza o valor anterior de c para o cálculo do erro na próxima iteração

    print(f"AVISO: max_iter={max_iter} atingido sem convergencia. Retornando melhor valor.")
    return c, historico

def newton(f, df, x0, eps=1e-8, max_iter=200):
    """ Retorna (raiz , historico ). """
    # Cria versões contadoras das funções f e df
    f_contada = contador(f)
    df_contada = contador(df)
    f = f_contada
    df = df_contada

    historico = []

    x_ant = x0
    fx_ant = f(x_ant) # Calcula f(x0) para a primeira iteração

    for i in range(1, max_iter + 1):
        dfx_ant = df(x_ant)

        # Verifica se a derivada é zero para evitar divisão por zero
        if dfx_ant == 0:
            raise ValueError(f"Derivada zero em x={x_ant}. Não é possível continuar.")

        # Fórmula de Newton
        x_atual = x_ant - (fx_ant / dfx_ant)

        # Calcula f(x_atual) para a próxima iteração
        fx_atual = f(x_atual)

        # Calcula o erro (igual como fizemos na bissecção)
        erro = abs(x_atual - x_ant)

        atual = {
            "k": i,
            "x": x_atual,
            "fx": fx_atual,
            "erro": erro,
            "chamadas_f": f_contada.n,
            "chamadas_df": df_contada.n,
            "convergiu": False
        }
        historico.append(atual)

        if abs(fx_atual) < eps or erro < eps:
            historico[-1]["convergiu"] = True
            return x_atual, historico

        # Atualiza os valores para a próxima iteração
        x_ant = x_atual
        fx_ant = fx_atual

    print(f"AVISO: max_iter={max_iter} atingido sem convergencia. Retornando melhor valor.")
    return x_atual, historico

def secante(f, x0, x1, eps=1e-8, max_iter=200):
    """ Retorna (raiz , historico ). """
    f_contada = contador(f)
    f = f_contada

    historico = []

    # Dois pontos iniciais
    f0 = f(x0)
    f1 = f(x1)

    for k in range(1, max_iter + 1):
        denom = f1 - f0
        if denom == 0:
            raise ValueError(f"Denominador zero em iteracao {k} (f(x1) - f(x0) = 0).")

        # Fórmula da secante para encontrar o próximo ponto
        x_novo = x1 - f1 * (x1 - x0) / denom

        # Avalia a função no novo ponto
        fx_novo = f(x_novo)

        # Calcula o erro
        erro = abs(x_novo - x1)

        atual = {
            "k": k,
            "x": x_novo,
            "fx": fx_novo,
            "erro": erro,
            "chamadas_f": f_contada.n,
            "convergiu": False
        }
        historico.append(atual)

        # Critério de parada
        if erro < eps or abs(fx_novo) < eps:
            historico[-1]["convergiu"] = True
            return x_novo, historico

        # Atualiza os pontos para a próxima iteração
        x0 = x1
        f0 = f1
        x1 = x_novo
        f1 = fx_novo

    print(f"AVISO: max_iter={max_iter} atingido sem convergencia. Retornando melhor valor.")
    return x_novo, historico