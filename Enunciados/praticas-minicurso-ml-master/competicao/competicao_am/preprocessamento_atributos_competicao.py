import pandas as pd
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems

def gerar_atributos_ator(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(df_treino,["ator_1","ator_2","ator_3","ator_4","ator_5"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(df_data_to_predict,["ator_1","ator_2","ator_3","ator_4","ator_5"])

    

    return df_treino_boa, df_data_to_predict_boa

def gerar_atributos_resumo(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    bow_amostra = BagOfWords()
    df_bow_treino = bow_amostra.cria_bow(df_treino,"resumo")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict,"resumo")

    return df_bow_treino,df_bow_data_to_predict
