import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# O objetivo deste código é implementar um Perceptron, um modelo de classificação linear simples,
# e usá-lo para classificar amostras do conjunto de dados Iris em duas classes.

class Perceptron(object):
    """
    Parâmetros:
    -----------
    eta: taxa de aprendizagem
    n_iter: número de passagens (épocas)
    
    Atributos
    ----------
    w_ : pesos após o ajuste
    errors_ : número de classificações incorretas em cada época
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Ajusta os dados de treinamento.

        Parâmetros
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Vetores de treinamento, onde n_samples é o número de amostras e
            n_features é o número de características.
        y : array-like, shape = [n_samples]
            Valores alvo.

        Retorna
        -------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print(f"Época {i+1}: erros = {errors}")
            print(f"Pesos atualizados: {self.w_}")
        return self

    def net_input(self, X):
        """Calcula a entrada líquida."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Retorna o rótulo de classe após a etapa de unidade."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Carregando o conjunto de dados Iris
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# Plotando as amostras de dados originais
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='Outras')
plt.xlabel('Comprimento da sépala [cm]')
plt.ylabel('Largura da sépala [cm]')
plt.legend(loc='upper left')
plt.title('Conjunto de dados Iris')
plt.show()

# Treinando o Perceptron
print("Iniciando o treinamento do Perceptron...")
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Salvando os pesos do Perceptron em um arquivo
print("Salvando os pesos do Perceptron...")
with open('iris_weights.pkl', 'wb') as f:
    pickle.dump(ppn.w_, f)
print("Os pesos do Perceptron foram salvos no arquivo 'iris_weights.pkl'.")

# Carregando os pesos do Perceptron a partir do arquivo
print("Carregando os pesos do Perceptron do arquivo...")
with open('iris_weights.pkl', 'rb') as f:
    perceptron_weights = pickle.load(f)

# Criando um novo objeto Perceptron com os pesos carregados
print("Criando um novo objeto Perceptron com os pesos carregados...")
ppn_loaded = Perceptron()
ppn_loaded.w_ = perceptron_weights

# Exemplo de como fazer previsões com o Perceptron carregado
print("Fazendo previsões com o Perceptron carregado...")
X_new = np.array([[5.1, 1.4], [4.9, 1.5]])  # Substitua pelos seus próprios dados de entrada
predictions = ppn_loaded.predict(X_new)
print("Previsões:", predictions)
