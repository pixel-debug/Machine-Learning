from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List
def cria_modelo(X:pd.DataFrame ,y:pd.Series ,min_samples_split:float):
    """
        Retorna o modelo a ser usado.

        X: matriz (ou DataFrame) em que cada linha é um exemplo e cada coluna é uma feature (atributo/caracteristica) do mesmo
        y: para cada posição i, y[i] é a classe alvo (ground truth) do exemplo x[i]
        min_samples_split: define o mínimo de exemplos necessários para que um nodo da árvore efetue a divisão.
                            Esse número pode ser uma porcentagm proporcional ao total de exemplos (quando float)
                            ou um número inteiro representando o número absoluto de exemplos.
    """
    #instancia a arvore de decisão (use  a classe DecisionTreeClassifier do Scikitlearn, defina o
    #..parametro min_samples_split e random_state=1 - é muito importante manter a seed fixa para
    #..que os sejam resultados mantenham sempre o mesmo - reprodutibilidade)
    decision_tree = None

    #retone o modelo por meio do método fit
    return None

def divide_treino_teste(df:pd.DataFrame, val_proporcao_treino:float) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
        A partir do DataFrame df, faz a divisão entre treino e teste obedecendo a proporção val_proporcao_treino.
        Essa proporção é um valor de 0 a 1, sendo que 1 representa 100%.

        Retorna uma tupla com o treino e teste separados
    """
    #1. obtenha o treino usando o método sample do DataFrame
    df_treino = None

    #2. Para obter o teste, selecione as instancias que estão em df e não estão em df_treino (use o método drop)
    df_teste = None

    return df_treino,df_teste



def faz_classificacao(x_treino:pd.DataFrame, y_treino:pd.Series, x_teste:pd.DataFrame, y_teste:pd.Series, min_samples_split:float) -> Tuple[List[float],float]:
    """
        Efetua a classificação, retornando:
            - O vetor y_predicted em que, para cada posição i,
             retorna o resultado previsto do exemplo representado
             por X_teste[i] que a classe alvo seria y_teste[i]. Esse y_predicted é
             o resultado retornado pelo método predict do modelo.
            - A acuracia (proporção de exemplos classificados corretamente)
                dicas:
                * caso tenhamos dois vetores a e b, ao fazer a operção a==b, ele retornará
                um vetor em que o valor  de cada posição i será igual a verdadeiro caso a==b.
                * np.sum soma os valores de um vetor (considerando True=1 e False=0)
    """
    #cria o modelo (use a função previamente criada)
    model_dtree = None

    #realiza a predição (use o método predict do modelo)
    y_predicted = None

    #calcule a acurácia
    acuracia = None


    return y_predicted,acuracia

def plot_performance_min_samples(X_treino,y_treino,X_teste,y_teste):
    """
        Crie um gráfico em que o eixo x é a variação do parametro min_sample e,
        o eixo y, representará a acurácia.
        Você deverá veriar o min_samples de 0.001 até 0.7 de 0.01 em 0.01 passos.
        Crie duas linhas: representando a acurácia no treino durante a variação do
        min_sample e, a outra, a acuracia do teste com os diversos valores de min_sample.
        Dicas:
            - A função arange do numpy pode ser usada no for (ao invés de range). Pois o range
            permite apenas passos com valores inteiros
            - para obter a acurácia no treino, o teste deverá possuir as mesmas instancias
            do treino
            - Entenda como é feito para plotar o grafico: https://matplotlib.org/users/pyplot_tutorial.html
    """
    #vetores que representam a acuracia no treino e no teste além do parametor usado
    arr_ac_treino = []
    arr_ac_teste = []
    arr_min_samples =[]

    for min_samples in np.arange(0.001,0.7,0.01):
        #complete a linha abaixo com a função e parametros corretos para calcular a acurácia no teste
        y_predicted, ac_teste = None
        #complete a linha abaixo com a função e parametros corretos para calcular a acurácia no treino
        y_predicted, ac_treino = None

        #adiciona a acuracia no treino, no teste e o parametro min_samples
        arr_ac_treino.append(ac_treino)
        arr_ac_teste.append(ac_teste)
        arr_min_samples.append(min_samples)

    #plota o resultado
    plt.plot(arr_min_samples,arr_ac_treino,"b--")
    plt.plot(arr_min_samples,arr_ac_teste,"r-")
