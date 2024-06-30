from copy import deepcopy
import numpy as np
import sys

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
    for j in range(len(matrizA)):
        linha = []
        for i in range(len(vetorX)):
            linha.append(matrizA[j][vetorX[i]])
        subMatriz.append(linha)
    return subMatriz

# Função para trocar linhas
def troca_linhas(matriz, i, j):
    matriz[i], matriz[j] = matriz[j], matriz[i]

# Função para calcular a inversa
def calcular_inversa(matrizBasica):
    def fimExecucao():
        print("Fim da Execução")
        sys.exit(0)

    def identidade(idt, k):
        for i in range(k):
            for j in range(k):
                if i == j:
                    idt[i][j + k] = 1
        return idt

    def matrizAumentada(matIni, k):
        m = np.zeros((k, 2 * k))
        for i in range(k):
            for j in range(k):
                m[i][j] = matIni[i][j]
        m = identidade(m, k)
        return m

    def gaussJordan(mat, k):
        for i in range(k):
            if mat[i][i] == 0.0:
                for p in range(i + 1, k):
                    if mat[p][i] != 0.0:
                        mat[[i, p]] = mat[[p, i]]
                        break
            for j in range(k):
                if i != j:
                    ratio = mat[j][i] / mat[i][i]
                    for l in range(2 * k):
                        mat[j][l] -= ratio * mat[i][l]
        for i in range(k):
            divisor = mat[i][i]
            for j in range(2 * k):
                mat[i][j] /= divisor
        return mat

    def divideMatriz(matriz, k):
        matR = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                matR[i][j] = matriz[i][j + k]
        return matR

    tam = len(matrizBasica)
    det = np.linalg.det(matrizBasica)

    if det != 0.0:
        matrizA = matrizAumentada(matrizBasica, tam)
        inversa = gaussJordan(matrizA, tam)
        final = divideMatriz(inversa, tam)
        return final
    else:
        print("Determinante Inválido -- inversa não possível")
        fimExecucao()

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
            if(cond == '<='):
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
            if (i == j):
                linha.append(ineq[j])
            else:
                linha.append(0.0)
        matrizA.append(linha)
    # Inicio da definicao de basicas e nao basicas
    basicas = []
    naoBasicas = []
    tam = len(funcaoObjetivo)
    for i in range(tam):
        if(i < tam-tamanho):
            naoBasicas.append(i)
        else:
            basicas.append(i)
    basicas.sort()
    naoBasicas.sort()
    # Inicio da criacao da matriz de termos independentes
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

# passo 3 teste de otimalidade
def verificar_otimo(custoRelativoNaoBasico, k):
    return custoRelativoNaoBasico[k] >= 0

# passo 4 calculo de direcao simplex
def direcao_simplex(basicaInversa, matrizA, k, naoBasicas):
    colunaK = [matrizA[i][naoBasicas[k]] for i in range(len(matrizA))]
    colunaK = np.transpose(colunaK)
    y = np.matmul(basicaInversa, colunaK)
    return y

# passo 5 determinacao do passo e variavel a sair da base
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

# passo 6 atualizacao nova particao basica trocando a l-esima coluna de B pela k-esima coluna de N
def troca_linhas_k_l(basicas, naoBasicas, k, l):
    basicas[l], naoBasicas[k] = naoBasicas[k], basicas[l]
    return basicas, naoBasicas

# calculo funcao final
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
            funcaoObjetivo[i] = -funcaoObjetivo[i]

    while it < maxit:
        print(f"\nIteração {it + 1}:")
        matrizBasica = calcular_submatriz(matrizA, basicas)
        matrizNaoBasica = calcular_submatriz(matrizA, naoBasicas)
        try:
            basicaInversa = calcular_inversa(matrizBasica)
        except np.linalg.LinAlgError:
            funciona = False
            break
        xRelativoBasico = calculo_relativo(basicaInversa, b)
        lambdaSimplex = calcula_lambda(
            calculo_custo(funcaoObjetivo, basicas), basicaInversa)
        custoRelativoNaoBasico = custos_relativos(
            lambdaSimplex,
            calculo_custo(funcaoObjetivo, naoBasicas), matrizNaoBasica)
        k = calcula_k(custoRelativoNaoBasico)
        if verificar_otimo(custoRelativoNaoBasico, k):
            otima = xRelativoBasico
            break
        y = direcao_simplex(basicaInversa, matrizA, k, naoBasicas)
        l = calcula_l(y, xRelativoBasico)
        if l is False:
            funciona = False
            break
        basicas, naoBasicas = troca_linhas_k_l(basicas, naoBasicas, k, l)
        it += 1
        print("Variáveis Básicas:", basicas)
        print("Variáveis Não Básicas:", naoBasicas)
        print("x Relativo Básico:", xRelativoBasico.flatten())
        print("Custos Relativos Não Básicos:", custoRelativoNaoBasico)

    print_resultado(funciona, otima, basicas, fx)

# Função para ler o arquivo de entrada
def ler_arquivo(filename):
    with open(filename, 'r') as file:
        tipoProblema = file.readline().strip()
        funcaoObjetivo = list(map(float, file.readline().strip().split()))
        restricoes = []
        for linha in file:
            restricao = linha.strip().split()
            coeficientes = list(map(float, restricao[:-2]))
            operador = restricao[-2]
            termoIndependente = float(restricao[-1])
            restricoes.append(coeficientes + [operador, termoIndependente])
    return tipoProblema, funcaoObjetivo, restricoes

# Função principal
def main():
    filename = 'problema.txt'
    tipoProblema, funcaoObjetivo, restricoes = ler_arquivo(filename)
    calculo_simplex(tipoProblema, funcaoObjetivo, restricoes)

if __name__ == "__main__":
    main()