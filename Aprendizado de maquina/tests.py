from sklearn.tree import DecisionTreeClassifier
import unittest
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings

from avaliacao import Experimento, OtimizacaoObjetivoArvoreDecisao,OtimizacaoObjetivoRandomForest

from resultado import Resultado,Fold
from metodo import ScikitLearnAprendizadoDeMaquina

class TestResultado(unittest.TestCase):
    y =         np.array([0,0,1,1,1,2,2,2,2,2,2,2,2])
    predict_y = np.array([0,1,1,2,2,1,2,1,2,0,2,2,1])
    y_zero = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    predict_y_zero = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])


    def test_macro_f1(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        prec = [1/2,1/5,4/6]
        rev = [1/2,1/3,4/8]
        f1_esp = [2*(prec[i]*rev[i])/(prec[i]+rev[i]) for i in range(len(prec))]

        macro_f1 = np.average(f1_esp)
        self.assertAlmostEqual(resultado.macro_f1,macro_f1,msg="Macro F1 não está com o valor esperado")
    def test_acuracia(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        self.assertAlmostEqual(resultado.acuracia,6/13,msg="Acuracia não está com o valor esperado")
class Dados:
    df_treino = pd.DataFrame({"A":[1, 1, 2, 2, 3, 4, 4, 5, 6, 1],
                    "B": [True,False,True,False,True,False,False,False,False,True],
                    "C":[23, 3, 123, 55, 12,33,44,21,55,22],
                    "D":[1, 1,  1, 1, 1, 1 , 1 , 1 , 1 , 1 ],
                    "realClass":[1,1,0,0,0,1,1,0,1,0]})

    df_teste = pd.DataFrame({"A":[1,1,1,2,3,3,3,3,4,4,4,4,5,5,5],
                    "B": [True,False,True,True,False,True,True,False,True,True,False,True,True,False,True],
                    "C":[333,-1,5,333,-12,52,3323,-12,52,3323,-41,53,3333,-12,51],
                    "D":[2, 2, 3,2, 2, 3,2, 23, 3,2, 21, 3,2, 22, 3],
                    "realClass":[1,0,1,1,0,1,1,0,1,1,0,0,0,0,1]})
    df_dados = pd.DataFrame({"A":[1, 1, 2, 2, 3, 4, 4, 5, 6,1,1,1,1,2,3,3,3,3,4,4,4,4,5,5,5],
                    "B": [True,False,True,False,True,False,False,False,False,True,
                         True,False,True,True,False,True,True,False,True,True,False,True,True,False,True],
                    "C":[23, 3, 123, 55, 12,33,44,21,55,22,333,-1,5,333,-12,52,3323,-12,52,3323,-41,53,3333,-12,51],
                    "D":[1,  1, 1, 1, 1, 1, 1, 1, 1,1,2, 2, 3,2, 2, 3,2, 23, 3,2, 21, 3,2, 22, 3],
                    "realClass":[1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,1,1,0,1,1,0,0,0,0,1]})

class MetodoTest(unittest.TestCase):
    def test_eval(self):
        clf_dtree = DecisionTreeClassifier(random_state=1)
        metodo = ScikitLearnAprendizadoDeMaquina(clf_dtree)
        resultado = metodo.eval(Dados.df_treino,Dados.df_teste,"realClass")

        self.assertListEqual(list(Dados.df_teste["realClass"]),list(resultado.y),"A lista de classe alvo da partição de teste não é a esperada")
        acuracia = resultado.acuracia
        macro_f1 = resultado.macro_f1
        print(f"Macro f1: {macro_f1} Acuracia: {acuracia}")

        self.assertAlmostEqual(macro_f1, 0.5982142857142857,msg="Macro F1 não está com o valor esperado")
        self.assertAlmostEqual(acuracia, 0.6,msg="Acuracia não está com o valor esperado")


class TestFold(unittest.TestCase):
    @staticmethod
    def folds_test(tester,df_dados,folds,k,is_cross_validation,num_repeticao):


        #verifica se a soma das instancias de teste=1
        lstTeste = set()
        for i,f in enumerate(folds):
            ids_teste = set(f.df_data_to_predict.index.values.tolist())
            ids_treino = set(f.df_treino.index.values.tolist())

            #verifica se o treino e teste possui algum item em comum
            itens_comuns = ids_teste & ids_treino
            tester.assertTrue(len(itens_comuns)==0,f"Existem instancias iguais no treino e na amostra para predição: {itens_comuns} no fold #{i} repeticao #{num_repeticao}")

            #verifica se o todas as instancias foram usadas
            tester.assertEqual(len(df_dados),len(ids_teste)+len(ids_treino),f"A soma do itens do treino e dos itens para predição não está igual ao dataset completo no fold #{i} repeticao #{num_repeticao}")

            #verifica se o teste nao foi usado em outro fold
            for j,fj in enumerate(folds):
                if(i!=j):
                    ids_teste_j = set(fj.df_data_to_predict.index.values.tolist())
                    itens_comuns = ids_teste & ids_teste_j
                    tester.assertTrue(len(itens_comuns)==0,f"Instancias no teste do fold {i} repeticao #{num_repeticao} também foi usado em  no fold {j}. Indices comuns:{itens_comuns}")

            lstTeste = lstTeste | set(f.df_data_to_predict.index.values.tolist())

            #verifica se o temanho do teste esta correto do dataset
            #if(i<k-1):
            #    tester.assertEqual(tam_fold,len(ids_teste),"O tamanho do partição deveria ser floor(numero_de_itens/val_k) - exceto o ultimo que deve possuir mais.")
            #else:
            #    tester.assertTrue(len(ids_teste)>=tam_fold, "No ultimo fold, o tamanho da particao deve ser maior ou igual a floor(numero_de_itens/val_k)")
        #verifica se todas as instancias ficaram em alguma particao de teste
        if(is_cross_validation):
            tester.assertEqual(len(lstTeste),len(df_dados),"Algumas instancias não foram usadas no teste.")

    def test_gerar_k_folds(self):
        k = 7
        num_repeticoes = 3

        #print("DADOS: "+str(len(TestFold.df_dados)))
        tam_fold = len(Dados.df_dados)//k
        folds = Fold.gerar_k_folds(Dados.df_dados,col_classe="realClass",val_k=k,num_repeticoes=num_repeticoes,seed=1)

        #verifica se foram 4 folds e 3 repetições
        self.assertEqual(k*num_repeticoes,len(folds),"O número de folds criado não é quantidade solicitada")

        #verifica se os dados estao embaralhados
        arr_lista_fold0 = list(folds[0].df_data_to_predict.index.values)
        self.assertTrue(arr_lista_fold0!=[0,1,2], "A lista não foi embaralhada!")
        self.assertListEqual(arr_lista_fold0,[14, 13, 17], "A lista não foi embaralhada corretamente! Não esqueça de usar a seed=seed+num_repeticoes")
        #verifica se os dados foram divididos corretamente

        #testa cada repetição separadamente
        for repeticao_i in range(num_repeticoes):
            folds_por_repeticao = folds[repeticao_i*k:repeticao_i*k+k]
            TestFold.folds_test(self,Dados.df_dados,folds_por_repeticao,k,True,repeticao_i)

            for i,f in enumerate(folds_por_repeticao):
                ids_teste = set(f.df_data_to_predict.index.values.tolist())
                ids_treino = set(f.df_treino.index.values.tolist())

                #verifica se o temanho do teste esta correto do dataset
                if(i<k-1):
                    self.assertEqual(tam_fold,len(ids_teste),"O tamanho do partição deveria ser floor(numero_de_itens/val_k) - exceto o ultimo que deve possuir mais.")
                else:
                    self.assertTrue(len(ids_teste)>=tam_fold, "No ultimo fold, o tamanho da particao deve ser maior ou igual a floor(numero_de_itens/val_k)")
    def test_arr_validacao(self):
        #testa a criação do fold
        fold = Fold(Dados.df_treino,Dados.df_teste,"realClass", num_folds_validacao=3,num_repeticoes_validacao=2)

        #verifica se foi criado 6 folds de validação
        self.assertEqual(len(fold.arr_folds_validacao),6,"Foi solicitado 2 execuções de 3 folds, ou seja, no final 6 folds")

        #os folds de validação nao possuem validação
        for fold_validacao in fold.arr_folds_validacao:
            self.assertEqual(len(fold_validacao.arr_folds_validacao),0,"O fold de validação não possuirá validação")

        #verifica cada execução
        arr_folds_execucao_1 = fold.arr_folds_validacao[:3]
        TestFold.folds_test(self,Dados.df_treino,arr_folds_execucao_1,3,True,1)

        arr_folds_execucao_2 = fold.arr_folds_validacao[3:]
        TestFold.folds_test(self,Dados.df_treino,arr_folds_execucao_2,3,True,1)




class ExperimentoTest(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
    def get_experimento(self,ml_method=DecisionTreeClassifier(min_samples_split=1,random_state=1),ClasseObjetivoOtimizacao=OtimizacaoObjetivoArvoreDecisao):


        folds = Fold.gerar_k_folds(Dados.df_dados,val_k=5,col_classe="realClass",
                                    num_repeticoes=1,seed=1,
                                    num_folds_validacao=3,num_repeticoes_validacao=2)

        exp = Experimento(folds,ml_method, ClasseObjetivoOtimizacao, num_trials=10,
                            sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=3))
        return exp

    def test_macro_f1_avg(self):

        exp = self.get_experimento()

        print("Macro F1 médio:"+str(exp.macro_f1_avg))
        self.assertAlmostEqual(exp.macro_f1_avg, 0.39380952380952383, msg="Valor inesperado de Macro F1")


    def test_resultados(self):

        exp = self.get_experimento()
        fold = exp.folds[0]

        arrExpMacroF1 =[0.16666666666666666,0.4444444444444444,
                        0.48888888888888893,0.6190476190476191, 0.24999999999999997]
        exp.calcula_resultados()

        for i,macro_f1 in enumerate(arrExpMacroF1):
            self.assertTrue(type(exp.resultados[i]) == Resultado, "O método calcula_resultados deve retornar uma lista de objetos da classe Resultado e não float.")
            print(f"Fold: {i} Macro F1: {exp.resultados[i].macro_f1}")
            #verifica se o melhor metodo foi usado


            #verifica se o resultado é o mesmo
            self.assertAlmostEqual(macro_f1,exp.resultados[i].macro_f1,msg=f"A Macro F1 do fold {i} não está com o valor esperado.")


class TestObjetivoOtimizacaoRF(unittest.TestCase):
    def test_otimizacao(self):
        fold = Fold(Dados.df_treino,Dados.df_teste,"realClass", num_folds_validacao=3,num_repeticoes_validacao=2)
        otimiza_fold = OtimizacaoObjetivoRandomForest(fold)
        tpe_sampler = TPESampler(n_startup_trials = 10,seed=1)
        study_TP = optuna.create_study(sampler=tpe_sampler, direction="maximize")
        study_TP.optimize(otimiza_fold, n_trials=30)

        for trial in study_TP.trials:
            print(trial.params)
        arr_params_to_test = ["min_samples_split", "max_features", "num_arvores"]
        for param_name in arr_params_to_test:
            self.assertTrue(param_name in study_TP.best_trial.params, f"Não foi encontrado o parametro '{param_name}' certifique-se se você nomeou o parametro devidamente")

        self.assertAlmostEqual(study_TP.best_trial.params["min_samples_split"],0.19829036364801306,places=5,msg="Otimização não deu resultado esperado")
        self.assertAlmostEqual(study_TP.best_trial.params["max_features"],0.1939553705810037,places=5,msg="Otimização não deu resultado esperado")
        self.assertAlmostEqual(study_TP.best_trial.params["num_arvores"],5,msg="Otimização não deu resultado esperado")
        print(f"Melhor execução: {study_TP.best_trial.params}")

        result = Resultado(np.array([1,1,1,1,0,0,0,0]),np.array([1,1,0,0,1,1,1,0]))
        result_metrica = otimiza_fold.resultado_metrica_otimizacao(result)
        print(f"Resultado: {result_metrica}")
        self.assertAlmostEqual(result_metrica,0.3650793650793651,places=5,msg="Resultado da metrica de otimização não deu resultado esperado")
if __name__ == "__main__":
    unittest.main()
