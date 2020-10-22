import unittest
import pandas as pd
from ganho_informacao import *

class TestInfoGain(unittest.TestCase):
    df_teste = df = pd.DataFrame({"A":["X", "X", "X", "S", "S","1","1","1"],
                    "B": [True,False,True,False,True,False,False,False],
                    "C":["X", "0", "X", "S", "S","1","0","S"],
                    "D":["X", "X", "X", "X", "X","X","X","X"]})

    def test_entropy(self):
        results ={"A":1.561,
                  "B":0.9544,
                  "C":1.9056,
                  "D":0}
        for col,entropia_esperada in results.items():
            self.assertAlmostEqual(entropia_esperada, #resultado esperado da feature na posicao i, vertcie na posicao posGrafo
                                    entropia(TestInfoGain.df_teste,col),#resultado obtido
                                    3,# numero de casas decimais que devem ser iguais nesse resultado
                                    )
    def teste_ganho_informacao_condicional(self):
        results ={
                "B":{
                True:0.6429,
                False:0.1903
                },
                "C":{
                "S":0.6429,
                "X":1.5612,
                "0":0.5612,
                "1":1.5612,
                },
                "D":{
                "X":0
                }
        }
        for col, dic in results.items():
            val_entropia_y = entropia(TestInfoGain.df_teste,"A")
            for val_atributo,val_gi in results[col].items():
                val_obtido = ganho_informacao_condicional(TestInfoGain.df_teste,
                                            val_entropia_y,
                                            "A",
                                            col,val_atributo)
                self.assertAlmostEqual(val_gi, #resultado esperado da feature na posicao i, vertcie na posicao posGrafo
                                        val_obtido,#resultado obtido
                                        3,# numero de casas decimais que devem ser iguais nesse resultado
                                        "GI(A| {at}={val}) deveria ser {esperado} mas foi {obtido}".format(at=col,
                                                                                                    val=val_atributo,
                                                                                                    esperado=val_gi,
                                                                                                    obtido=val_obtido)
                                        )
    def test_info_gain(self):
        results ={
                  "B":0.3600,
                  "C":0.9669,
                  "D":0
                  }
        for col,gi_esperado in results.items():
                #print(ganho_informacao(TestInfoGain.df_teste,"A",col))
                val_obtido = ganho_informacao(TestInfoGain.df_teste,"A",col)
                self.assertAlmostEqual(gi_esperado, #resultado esperado da feature na posicao i, vertcie na posicao posGrafo
                                        val_obtido,#resultado obtido
                                        3,# numero de casas decimais que devem ser iguais nesse resultado
                                        "GI(A| {at}) deveria ser {esperado} mas foi {obtido}".format(at=col,
                                                                                                    esperado=gi_esperado,
                                                                                                    obtido=val_obtido)
                                        )
if __name__ == "__main__":
    unittest.main()
