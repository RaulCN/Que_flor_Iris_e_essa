# Que_flor_Iris_e_essa-
Repositório criado para upar datasets, scripts e modelos utilizados no problema clássico de classificação de flores do conjunto de dados flor Iris


Descrição dos scripts

perceptroniris.py (faz o treinamento simples e classifica de forma binária as flores entre "Iris-Setosa" e "outros"

perceptronirisplot.py (arquivo igual o primeiro mas com uma plotagem do resultado)

perceptronirispesoseprevisao.py (nesse é feito o treinamento, é os pesos são salvos e posteriormente são carregados para fazer previsões, tbm é feito uma plotagem do resultado inicial) 

iris_weights.pkl (arquivo com pesos de treinamento) 

perceptronirisplotprevisaocompesos.py (e por fim um código mais sofisticado carregando os pesos para ir mais rapido com entrada de novos dados para previsao) 

Info_Iris_sklearn.py (gera uma explicação do conjunto de dados presentes no sklearn) veja a explicação completa:

Descrição do conjunto de dados Iris:
Esta parte apresenta a descrição do conjunto de dados Iris. Essa descrição fornece informações gerais sobre o conjunto de dados, como o número de instâncias, o número de atributos, a distribuição das classes, entre outras.

Nomes das características (atributos):
Aqui são listados os nomes das características (atributos) presentes no conjunto de dados Iris. No caso do conjunto de dados Iris, há quatro características: "sepal length (cm)", "sepal width (cm)", "petal length (cm)" e "petal width (cm)".

Nomes das classes alvo:
Esta parte lista os nomes das classes alvo presentes no conjunto de dados Iris. Existem três classes alvo: "setosa", "versicolor" e "virginica". Essas classes representam diferentes espécies de flores Iris.

Dados das características (atributos):
Aqui são exibidos os dados das características (atributos) do conjunto de dados Iris. Cada linha corresponde a uma instância do conjunto de dados, e cada coluna corresponde a um atributo. Portanto, os valores exibidos são as medidas das características das flores Iris.

Rótulos das classes alvo:
Esta parte mostra os rótulos das classes alvo para cada instância do conjunto de dados Iris. Cada valor corresponde ao rótulo da classe alvo associada à respectiva instância. Os rótulos são representados numericamente, onde 0 corresponde à classe "setosa", 1 à classe "versicolor" e 2 à classe "virginica".

Essa saída fornece uma visão geral dos dados presentes no conjunto de dados Iris, incluindo informações sobre suas características, classes alvo e rótulos associados. Essas informações são úteis para entender a estrutura do conjunto de dados antes de aplicar técnicas de aprendizado de máquina.


info_Iris_plots_sklearn.py
Esse script carrega o conjunto de dados Iris, plota um gráfico de dispersão para o comprimento e largura da sépala e um gráfico de dispersão para o comprimento e largura da pétala. Também plota histogramas para o comprimento da sépala e da pétala.

classificador_de_flores.py
Esboço de classificador de flores usando interface gráfica com TKINTER

