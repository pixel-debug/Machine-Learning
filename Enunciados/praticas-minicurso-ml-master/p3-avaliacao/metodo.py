from abc import abstractmethod
from resultado import Resultado
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from typing import List,Union
class MetodoAprendizadoDeMaquina:

    @abstractmethod
    def eval(self,df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str) -> Resultado:
        raise NotImplementedError

class ScikitLearnAprendizadoDeMaquina(MetodoAprendizadoDeMaquina):
    #Dica: Union é usado quando um parametro pode receber mais de um tipo
    #neste caso, ml_method é um ClassifierMixin ou RegressorMixin
    #essas duas classes são superclasses dos classficadores e métodos de regressão
    def __init__(self,ml_method:Union[ClassifierMixin,RegressorMixin]):
        self.ml_method = ml_method

    def eval(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, seed:int=1) -> Resultado:
        #Atividade 2: Implementação da classe eval - veja passo a passo

        #a partir de self.df_treino, separe os atributos  da classe
        #x_treino deverá ser um dataframe que possua apenas as colunas dos atributos (use o método drop com o parametro axis)
        #y_treino deverá possuir apenas os valores coluna da classe
        x_treino = None
        y_treino = None



        #execute o método fit  de ml_method e crie o modelo
        model = None
        #faça a mesma separação que fizemos em x_treino e y_treino nos dados a serem previstos
        x_to_predict = None
        y_to_predict = None

        #Impressao do x e y para testes
        #print("X_treino: "+str(x_treino))
        #print("y_treino: "+str(y_treino))
        #print("X_to_predict: "+str(x_to_predict))
        #print("y_to_predict: "+str(y_to_predict))

        #retorne o resultado por meio do método predict
        y_predictions = None
        return Resultado(None,None)
