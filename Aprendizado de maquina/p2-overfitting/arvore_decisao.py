from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

def cria_modelo(x,y,min_samples):
 
    decision_tree = None
    decision_tree = DecisionTreeClassifier(min_samples_split = min_samples, random_state = 1)
   
    return decision_tree.fit(x,y)

def divide_treino_teste(df,val_proporcao_treino):
      
    df_treino = df.sample(frac = val_proporcao_treino, random_state=1)
    df_teste = df.drop(df_treino.index)
   
    return df_treino,df_teste



def faz_classificacao(x_treino,y_treino,x_teste,y_teste,min_samples):
   
    model_dtree = cria_modelo(x_treino, y_treino, min_samples)
    y_predicted = model_dtree.predict(x_teste)
    acuracia = (np.sum(y_predicted == y_teste)/len(y_teste))


    return y_predicted,acuracia

def plot_performance_min_samples(X_treino,y_treino,X_teste,y_teste):
    
    arr_ac_treino = []
    arr_ac_teste = []
    arr_min_samples =[]
    for min_samples in np.arange(0.001,0.7,0.01):
        y_predicted, ac_teste = faz_classificacao(X_treino,
                                                  y_treino,
                                                  X_treino,
                                                  y_treino,min_samples
                                                 )
        
        y_predicted, ac_treino = faz_classificacao(X_treino,
                                                   y_treino,
                                                   X_teste,
                                                   y_teste,min_samples
                                                  )

       
        arr_ac_treino.append(ac_treino)
        arr_ac_teste.append(ac_teste)
        arr_min_samples.append(min_samples)
        
    plt.plot(arr_min_samples,arr_ac_treino,"b--")
    plt.plot(arr_min_samples,arr_ac_teste,"r-")
