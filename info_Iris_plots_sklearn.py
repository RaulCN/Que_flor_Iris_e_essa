import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Plotar informações básicas do conjunto de dados
plt.figure(figsize=(10, 6))

# Plotar comprimento da sépala versus largura da sépala
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Largura da Sépala (cm)')
plt.title('Comprimento da Sépala vs Largura da Sépala')

# Plotar comprimento da pétala versus largura da pétala
plt.subplot(2, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis')
plt.xlabel('Comprimento da Pétala (cm)')
plt.ylabel('Largura da Pétala (cm)')
plt.title('Comprimento da Pétala vs Largura da Pétala')

# Plotar histograma das características
plt.subplot(2, 2, 3)
plt.hist(X[:, 0], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Frequência')
plt.title('Histograma do Comprimento da Sépala')

plt.subplot(2, 2, 4)
plt.hist(X[:, 2], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Comprimento da Pétala (cm)')
plt.ylabel('Frequência')
plt.title('Histograma do Comprimento da Pétala')

plt.tight_layout()
plt.show()
