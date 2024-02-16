from sklearn.datasets import load_iris

# Carregar o conjunto de dados Iris
iris = load_iris()

# Exibir informações sobre o conjunto de dados
print("Descrição do conjunto de dados Iris:")
print(iris.DESCR)
print("")

# Exibir os nomes das características (atributos)
print("Nomes das características (atributos):")
print(iris.feature_names)
print("")

# Exibir os nomes das classes alvo
print("Nomes das classes alvo:")
print(iris.target_names)
print("")

# Exibir os dados das características (atributos)
print("Dados das características (atributos):")
print(iris.data)
print("")

# Exibir os rótulos das classes alvo
print("Rótulos das classes alvo:")
print(iris.target)
