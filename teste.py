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
def calcular_submatriz(matrizA : list, vetorX : list) -> float:
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
def calcular_inversa(matrizA: list):
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
def separa_matriz(funcaoObjetivo: list, restricoes : list) -> list:
    # Definicao das variaveis de folga
    ineq = []
    tamanho = len(restricoes)
    for i in range(tamanho):
        cond = restricoes[i][-2]
        if cond!= '=':
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
    #Inicio da definicao de basicas e nao basicas
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
    # Inicio da criaçao da matriz de termos independentes
    b = []
    for i in range(tamanho):
        b.append(restricoes[i][-1])
    #Restricao B tem elemento menor que 0
    for i in range (len(b)):
        if b[i] < 0:
            for j in range(len(matrizA[i])):
                matrizA[i][j] = -matrizA[i][j]
            b[i] = -b[i]
    return matrizA, basicas, naoBasicas, b

# Fase II:

#Passo 1 (calculo da solucao basica)

def calculo_relativo(BInv : list, b : list) -> float:
    return np.matmul(BInv, np.transpose(np.matrix(b)))

#Passo 2 (calculo dos custos relativos)
# 2.1 vetor multiplicador simplex

def calculo_custo(funcaoObjetivo : list, variaveis : list) -> float:
    custoBasico = [0] * len(variaveis)
    for i in range(len(custoBasico)):
        custoBasico[i] = funcaoObjetivo[variaveis[i]]
    return custoBasico

def calcula_lambda(custoBasico : list, basicaInversa:list) -> float:
    return np.matmul(custoBasico, basicaInversa)

# 2.2 custos relativos

def custos_relativos(lambdaSimplex: list, custoNaoBasico:list, matrizNaoBasica:list) -> float:
    naoBasicaTransposta = np.transpose(matrizNaoBasica)
    for i in range(len(custoNaoBasico)):
        custoNaoBasico[i] -= (np.dot(lambdaSimplex, naoBasicaTransposta[i]))
    return custoNaoBasico

# 2.3 determinaçao da variavel a entrar na base
def calcula_k(custoRelativoNaoBasico:list)-> int:
    return custoRelativoNaoBasico.index(min(custoRelativoNaoBasico))

# passo 3 teste de otimalidade
def verificar_otimo(custoRelativoNaoBasico: list, k : int) -> bool:
    return custoRelativoNaoBasico[k] >= 0

# passo 4 calculo de direcao simplex
def direcao_simplex(basicaInversa:list, matrizA:list, k:int, naoBasicas:list)->float:
    colunaK = [matrizA[i][naoBasicas[k]] for i in range(len(matrizA))]
    colunaK = np.transpose(colunaK)
    y = np.matmul(basicaInversa, colunaK)
    return y

# passo 5 determinacao do passo e variavel a sair da base
def calcula_l(y: list, xRelativoBasico:list) -> int:
    seguro = any(y[i] > 0 for i in range(len(y)))
    if not seguro:
        return False
    razoes =[
        xRelativoBasico[i] / y[i] if y[i] > 0 else float('inf')
        for i in range(len(xRelativoBasico))
    ]
    passo = min(razoes)
    l = razoes.index(passo)
    return l

# passo 6 atualizacao nova particao basica trocando a l-esima coluna de B pela k-esima coluna de N
def troca_linhas_k_l(basicas : list, naoBasicas : list, k: int, l : int) -> list:
    basicas[l], naoBasicas[k] = naoBasicas[k], basicas[l]
    return basicas, naoBasicas

# calculo funcao final
def calculo_funcaoZ(funcaoObjetivo : list, xRelativoBasico:list, basicas:list) -> float:
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

        print('Matriz Básica: ')
        print_matrizes(matrizBasica)

        print('Matriz Não Básica: ')
        print_matrizes(matrizNaoBasica)

        matrizBasicaInversa = calcular_inversa(matrizBasica)

        print('Matriz Básica Inversa: ')
        print(matrizBasicaInversa)

        if matrizBasicaInversa is False:
            funciona = False
            break

        xRelativo = calculo_relativo(matrizBasicaInversa, b)

        custoBasico = calculo_custo(funcaoObjetivo, basicas)
        print('Custo basico: ' , custoBasico)

        lambdaTransposta = calcula_lambda(custoBasico, matrizBasicaInversa)

        custoNaoBasico = calculo_custo(funcaoObjetivo, naoBasicas)
        print("Custo nao basica: ", custoNaoBasico)

        custoRelativoNaoBasico = custos_relativos(lambdaTransposta, custoNaoBasico, matrizNaoBasica)

        print('Custo relativo nao basico: ', custoRelativoNaoBasico)
        k = calcula_k(custoRelativoNaoBasico)
        if verificar_otimo(custoRelativoNaoBasico, k):
            print('\nOtimo!')
            otima = xRelativo
            funciona = True
            break
        print('\nNao otimo!')
        y = direcao_simplex(matrizBasicaInversa, matrizA, k, naoBasicas)
        l = calcula_l(y, xRelativo)
        if isinstance(l, bool) and l is False:
            funciona = False
            break
        basicas, naoBasicas = troca_linhas_k_l(basicas, naoBasicas, k, l)
        it += 1
    print_resultado(funciona, otima, basicas, fx)

if __name__ == "__main__":
    '''tipoProblema = 'min'
    funcaoObjetivo = [-5,-2]
    restricoes = [
        [7, -5, '<=', 13],
        [3, 2, '<=', 17],
        [0, 1, '<=', 2],
        [1, 0, '>=', 4], 
        # [1, 0,'<=',3],
    ]'''
    '''
    # 5.7 a)
    tipoProblema = 'max'
    funcaoObjetivo = [1, 1]
    restricoes = [
        [2, 1, '<=', 18],
        [-1, 2, '<=', 4],
        [3, -6, '<=', 12],
    ]
    '''
    
    # 5.7 b)
    tipoProblema = 'max'
    funcaoObjetivo = [6, 2]
    restricoes = [
        [3, 1, '<=', 33],
        [1, 1, '<=', 13],
    ]
    
    
    '''# 5.7 f)
    tipoProblema = 'max'
    funcaoObjetivo = [-1, 2]
    restricoes = [
        [-2, 1, '<=', 3],
        [3, 4, '<=', 5],
        [1, -1, '<=', 2],
    ]'''
    
    '''tipoProblema = 'max'
    funcaoObjetivo = [1, -1, 2]
    restricoes = [
        [1, 1, 1, '=', 3],
        [2, -1, 3, '<=', 4],
    ]'''

    print("\nCalculo 1")
    calculo_simplex(tipoProblema, funcaoObjetivo, restricoes)