# Machine-Learning
</br>

</br>
<p align="center">
  <img src="https://super.abril.com.br/wp-content/uploads/2016/09/super_imggato_digitando_0.gif" width="350">
</p>


<!--ts-->   
   * [Primeira parte](#primeira-parte)
   * [Segunda parte](#segunda-parte)
   * [Programação orientada a objetos](#programacao-orientada-a-objetos)    
<!--te-->

</br>

Primeira parte:
============
>  Análise de Atributos para Diferenciar Espécies de Plantas do Gênero Iris
</br>
Plantas do gênero Iris possuem diversas espécies que podem ser diferenciadas por algumas caracteristicas da flor. Nesta prática, iremos investigar quais atributos distinguem melhor algumas espécies dessa planta. Para isso, usaremos este dataset que possui 150 plantas do gênero Iris com atributos de sua flor (propriedades):</br>

* Tamanho e largura do cálice (em cm)</br>

* Tamanho e largura da pétala (em cm)</br>

Existem 3 espécies de plantas do genero Iris na base que serão usadas: Iris Setosa, Iris Virginifica e Iris Versicolor</br>

Roteiro da prática:</br>

* Calcular do InfoGain de cada atributo. Armazene o resultado em um DataFrame de duas colunas - nome do atributo e valor de infogain - ordene essa tabela pelo InfoGain.

* Gerar um gráfico de disperção (scatter plot) em que o eixo x e y são os dois atributos com InfoGain mais altos e com 3 grupos, cada grupo, uma espécie de flor diferente.</br>




Segunda parte:
============
> Impacto do Overfitting/Underfitting - Estimativa Automática da Qualidade de Conteúdo
Nesta prática, foram usados dados de 3.294 artigos da Wikipédia rotulados manualmente quanto a sua qualidade.</br>

Esses artigos passaram por uma avaliação pela comunidade de editores da Wikipedia. Tais editores classificaram esses artigos quanto a qualidade da seguinte forma:</br>

* Artigo Destaque (FA): Os artigos atribuídos a esta classe são, de acordo com os avaliadores, os melhores artigos da Wikipédia.
* Classe A (AC): os artigos da Classe A são considerados completos, mas com alguns problemas pendentes que precisam ser resolvidos para serem promovidos a Artigos em destaque.
* Artigo Bons (GA): Bons Artigos são aqueles sem problemas de lacunas ou conteúdo excessivo. Essas são boas fontes de informação, embora outras enciclopédias possam fornecer um conteúdo melhor.
* Classe B (BC): os artigos atribuídos a essa classe são considerados úteis para a maioria dos usuários, mas carecem de informações mais precisas.
* Classe Inicial (ST): os artigos da Classe Inicial ainda estão incompletos, embora contenham referências e ponteiros para informações mais completas.
* Artigos Rascunhos (SB): os artigos de toco são artigos de rascunho, com poucos parágrafos. Eles também têm poucas ou nenhumas citações. </br>
* 
Assim, Dalip et. al. (2009) fizeram o preprocessamento desses artigos para serem extraídos indicadores de qualidades tais como: idade do artigo, tamanho, número de citações. Com tais indicadores e a classe de qualidade, foi possível realizar a predição automática de qualidade de artigos da Wikipédia.</br>

Foi feita uma previsão da qualidade usando os indicadores proposto por Dalip et. al. (2009) e uma árvore de decisão.


Programação orientada a objetos:
============
Práticas para testar os conhecimento em programação orientada a objetos na linguagem python.
