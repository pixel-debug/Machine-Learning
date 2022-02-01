import unittest
import pandas as pd
from arvore_decisao import *
class TestDecisionTree(unittest.TestCase):
        df_X_treino = pd.DataFrame({"A":["1", "1", "2", "2", "3","4","4","5","6","1"],
                        "B": [True,False,True,False,True,False,False,False,False,True],
                        "C":[23, 3, 123, 55, 12,33,44,21,55,22],
                        "D":["1", "1", "1", "1", "1","1","1","1","1","1"],
                        #"realClass":["1","22","5","44","2","1","2","2","3","3"]}
                        })
        df_y_treino = ["X","X","X","Y","Y","Z","Z","Z","X","X"]

        df_X_teste = pd.DataFrame({"A":[2, 3,1],
                        "B": [True,False,True],
                        "C":[333,-1,5],
                        "D":[1, 1, 1]})
        df_y_teste = ["X","Y","Z"]

        def test_cria_modelo(self):
            self.assertIsNotNone(cria_modelo(TestDecisionTree.df_X_treino,TestDecisionTree.df_y_treino,0.8))

        def test_divide_treino_teste(self):
            #testa varias divisoes
            for i in range(1,9):
                df_treino, df_teste = divide_treino_teste(TestDecisionTree.df_X_treino,i/10)
                self.assertEqual(len(df_treino),i,"O teste não está com a quantidade correta de elementos")
                self.assertEqual(len(df_teste),10-i,"O teste não está com a quantidade correta de elementos")
            #para uma divisao testa os elementos exatos
            df_treino, df_teste = divide_treino_teste(TestDecisionTree.df_X_treino,0.8)
            self.assertListEqual(list(df_treino["C"]),[123, 22, 44, 12, 23, 55, 3, 21],"O treino não está com os valores previstos")
            self.assertListEqual(list(df_teste["C"]), [33,55],"O teste não está com os valores previstos")

        def test_faz_classificacao(self):
            arr_predicted, acuracia = faz_classificacao(TestDecisionTree.df_X_treino,
                                                        TestDecisionTree.df_y_treino,
                                                        TestDecisionTree.df_X_teste,
                                                        TestDecisionTree.df_y_teste,
                                                        0.1)
            self.assertListEqual(list(arr_predicted),["X","Y","X"],"A predição não deu o resultado esperado")
            self.assertAlmostEqual(acuracia,0.6666666666)
if __name__ == "__main__":
    unittest.main()
