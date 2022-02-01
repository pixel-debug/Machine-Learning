from competicao_am.metodo_competicao import MetodoCompeticao
from sklearn.svm import LinearSVC
import pandas as pd
def gerar_saida_teste( df_data_to_predict, col_classe, num_grupo):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """

    scikit_method = LinearSVC(C=7, random_state=2)
    ml_method = MetodoCompeticao(scikit_method)
    
    #o treino será sempre o dataset completo - sem nenhum dado a mais e sem nenhum preprocessamento
    #esta função que deve encarregar de fazer o preprocessamento
    df_treino = pd.read_csv("datasets/movies_amostra.csv")

    #gera as representações e seu resultado
    arr_predictions_ator, y_to_predict = ml_method.eval_actors(df_treino, df_data_to_predict, col_classe)
    arr_predictions_bow, y_to_predict = ml_method.eval_bow(df_treino, df_data_to_predict, col_classe)
    
    #combina as duas
    arr_final_predictions = ml_method.combine_predictions(arr_predictions_ator, arr_predictions_bow)

    #grava o resultado obtido
    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in arr_final_predictions:
            file_predict.write(ml_method.dic_int_to_nom_classe[predict]+"\n")
