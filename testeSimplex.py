from copy import deepcopy
import numpy as np

# Função mostrar matriz
def print_matrizes(mat):
    for linha in mat:
        print(linha)

# Função para imprimir o resultado
def print_resultado(funciona, otima, basicas, fx):
    if funciona:
        print("A solução factível ótima é:")
        resultado = calculo_funcaoZ(fx, otima, basicas)
        print(', '.join(
            [f'x{basicas[i]} = {otima[i]}'
                for i in range(len(otima))]) + f', z = {resultado}')
        print("\n Finalizado!")
    else:
        print("Em algum momento não foi possível fazer a inversa ou a direção simplex é <= 0.")
        print("Finalizado!")

# Função para calcular as submatrizes básicas e não básicas
def calcular_submatriz(matrizA, vetorX):
    subMatriz = []
    for j in range (len(matrizA)):
        linha = []
        for i in range(len(vetorX)):
            linha.append(matrizA[j][vetorX[i]])
        subMatriz.append(linha)
    return subMatriz

# Função para trocar linhas
def troca_linhas(matriz, i, j):
    matriz[i], matriz[j] = matriz[j], matriz[i]

# Função para calcular a inversa 
def calcular_inversa(matrizA):
    n = len(matrizA)
    I = np.identity(n)
    det = np.linalg.det(matrizA)
    if det != 0:
        AID = np.hstack((matrizA, I))

        for i in range(n):
            pivo = AID[i][i]

            if pivo == 0:
                for j in range(i + 1, n):
                    if AID[j, i] != 0:
                        AID[[i, j]] = AID[[j, i]]
                        break
                else:
                    print('Matriz não possui inversa')
                    return None

                pivo = AID[i, i]

            AID[i] /= pivo

            for j in range(n):
                if j != i:
                    multiplicador = AID[j, i]
                    AID[j] -= multiplicador * AID[i]

        inversa = AID[:, n:]
        return inversa
    else:
        print('Matriz não possui inversa')
        return None


# Simplex
# Fase I
def separa_matriz(funcaoObjetivo, restricoes):
    # Definicao das variaveis de folga
    ineq = []
    tamanho = len(restricoes)
    for i in range(tamanho):
        cond = restricoes[i][-2]
        if cond != '=':
            funcaoObjetivo.append(0.0)
            if cond == '<=':
                ineq.append(1.0)
            else:
                ineq.append(-1.0)
        else:
            ineq.append(0.0)
    # Inicio do aumento da matriz A
    matrizA = []
    for i in range(tamanho):
        linha = restricoes[i][:-2]
        tamanho2 = len(ineq)
        for j in range(tamanho2):
            if i == j:
                linha.append(ineq[j])
            else:
                linha.append(0.0)
        matrizA.append(linha)
    # Inicio da definicao de basicas e nao basicas
    basicas = []
    naoBasicas = []
    tam = len(funcaoObjetivo)
    for i in range(tam):
        if i < tam - tamanho:
            naoBasicas.append(i)
        else:
            basicas.append(i)
    basicas.sort()
    naoBasicas.sort()
    # Inicio da criaçao da matriz de termos independentes
    b = []
    for i in range(tamanho):
        b.append(restricoes[i][-1])
    # Restricao B tem elemento menor que 0
    for i in range(len(b)):
        if b[i] < 0:
            for j in range(len(matrizA[i])):
                matrizA[i][j] = -matrizA[i][j]
            b[i] = -b[i]
    return matrizA, basicas, naoBasicas, b

# Fase II:

# Passo 1 (calculo da solucao basica)
def calculo_relativo(BInv, b):
    return np.matmul(BInv, np.transpose(np.matrix(b)))

# Passo 2 (calculo dos custos relativos)
# 2.1 vetor multiplicador simplex
def calculo_custo(funcaoObjetivo, variaveis):
    custoBasico = [0] * len(variaveis)
    for i in range(len(custoBasico)):
        custoBasico[i] = funcaoObjetivo[variaveis[i]]
    return custoBasico

def calcula_lambda(custoBasico, basicaInversa):
    return np.matmul(custoBasico, basicaInversa)

# 2.2 custos relativos
def custos_relativos(lambdaSimplex, custoNaoBasico, matrizNaoBasica):
    naoBasicaTransposta = np.transpose(matrizNaoBasica)
    for i in range(len(custoNaoBasico)):
        custoNaoBasico[i] -= (np.dot(lambdaSimplex, naoBasicaTransposta[i]))
    return custoNaoBasico

# 2.3 determinacao da variavel a entrar na base
def calcula_k(custoRelativoNaoBasico):
    return custoRelativoNaoBasico.index(min(custoRelativoNaoBasico))

# Passo 3 teste de otimalidade
def verificar_otimo(custoRelativoNaoBasico, k):
    return custoRelativoNaoBasico[k] >= 0

# Passo 4 calculo de direcao simplex
def direcao_simplex(basicaInversa, matrizA, k, naoBasicas):
    colunaK = [matrizA[i][naoBasicas[k]] for i in range(len(matrizA))]
    colunaK = np.transpose(colunaK)
    y = np.matmul(basicaInversa, colunaK)
    return y

# Passo 5 determinacao do passo e variavel a sair da base
def calcula_l(y, xRelativoBasico):
    seguro = any(y[i] > 0 for i in range(len(y)))
    if not seguro:
        return False
    razoes = [
        xRelativoBasico[i] / y[i] if y[i] > 0 else float('inf')
        for i in range(len(xRelativoBasico))
    ]
    passo = min(razoes)
    l = razoes.index(passo)
    return l

# Passo 6 atualizacao nova particao basica trocando a l-esima coluna de B pela k-esima coluna de N
def troca_linhas_k_l(basicas, naoBasicas, k, l):
    basicas[l], naoBasicas[k] = naoBasicas[k], basicas[l]
    return basicas, naoBasicas

# Calculo funcao final
def calculo_funcaoZ(funcaoObjetivo, xRelativoBasico, basicas):
    resultado = sum(funcaoObjetivo[basicas[i]] * xRelativoBasico[i] for i in range(len(xRelativoBasico)))
    return resultado

# Simplex
def calculo_simplex(tipoProblema, funcaoObjetivo, restricoes):
    it = 0
    maxit = 25
    otima = []
    funciona = True
    matrizA, basicas, naoBasicas, b = separa_matriz(funcaoObjetivo, restricoes)
    matrizA = np.array(matrizA)
    b = np.array(b)
    
    fx = deepcopy(funcaoObjetivo)
    tamanho = len(funcaoObjetivo)
    if tipoProblema == 'max':
        for i in range(tamanho):
            funcaoObjetivo[i] *= -1
    while it < maxit:
        print(f'\niteracao: {it+1}')
        matrizBasica = calcular_submatriz(matrizA, basicas)
        matrizNaoBasica = calcular_submatriz(matrizA, naoBasicas)

        print('MatrizA: ')
        print_matrizes(matrizA)
        print('\nBasicas: ', basicas)
        print('Matriz basica: ')
        print_matrizes(matrizBasica)
        print('\nNao basicas: ', naoBasicas)
        print('Matriz nao basica: ')
        print_matrizes(matrizNaoBasica)
        print('\nTermos independentes: ', b)

        matrizBasicaInversa = calcular_inversa(matrizBasica)
        if matrizBasicaInversa is None:
            print('Erro ao calcular a inversa da matriz básica')
            funciona = False
            break

        xRelativoBasico = calculo_relativo(matrizBasicaInversa, b)
        print('\nRelativo: ', xRelativoBasico)

        custoBasico = calculo_custo(funcaoObjetivo, basicas)
        print('Custos basicos: ', custoBasico)

        lambdaSimplex = calcula_lambda(custoBasico, matrizBasicaInversa)
        print('Lambda: ', lambdaSimplex)

        custoNaoBasico = calculo_custo(funcaoObjetivo, naoBasicas)
        print('Custos nao basicos: ', custoNaoBasico)

        custoRelativoNaoBasico = custos_relativos(lambdaSimplex, custoNaoBasico, matrizNaoBasica)
        print('Custos relativos: ', custoRelativoNaoBasico)

        k = calcula_k(custoRelativoNaoBasico)
        print('Valor de k: ', k)
        if verificar_otimo(custoRelativoNaoBasico, k):
            print('Otima: ', xRelativoBasico)
            otima = xRelativoBasico
            break

        direcao = direcao_simplex(matrizBasicaInversa, matrizA, k, naoBasicas)
        print('Direcao: ', direcao)

        l = calcula_l(direcao, xRelativoBasico)
        if l is False:
            funciona = False
            break

        print('Valor de l: ', l)

        basicas, naoBasicas = troca_linhas_k_l(basicas, naoBasicas, k, l)
        it += 1

    print_resultado(funciona, otima, basicas, fx)

def ler_arquivo(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    tipoProblema = lines[0].strip()
    funcaoObjetivo = list(map(float, lines[1].strip().split()))
    restricoes = []

    for line in lines[2:]:
        parts = line.strip().split()
        coeficientes = list(map(float, parts[:-2]))
        sinal = parts[-2]
        valor = float(parts[-1])
        restricao = coeficientes + [sinal, valor]
        restricoes.append(restricao)

    return tipoProblema, funcaoObjetivo, restricoes

if __name__ == "__main__":
    filename = 'problema.txt'
    tipoProblema, funcaoObjetivo, restricoes = ler_arquivo(filename)
    calculo_simplex(tipoProblema, funcaoObjetivo, restricoes)
