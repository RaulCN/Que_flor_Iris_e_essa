import numpy as np
import pandas as pd
import pickle

# Carregando os pesos do Perceptron a partir do arquivo
with open('iris_weights.pkl', 'rb') as f:
    perceptron_weights = pickle.load(f)

# Criando um objeto Perceptron com os pesos carregados
class PerceptronPrediction(object):
    def __init__(self):
        self.w_ = perceptron_weights

    def net_input(self, X):
        """Calcula a entrada líquida."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Retorna o rótulo de classe após a etapa de unidade."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Função para prever a classe da flor com base nos dados de entrada
def predict_flower(sepal_length, sepal_width):
    # Criando um objeto Perceptron para fazer previsões
    ppn = PerceptronPrediction()
    # Preparando os dados de entrada
    X_new = np.array([[sepal_length, sepal_width]])
    # Fazendo a previsão
    prediction = ppn.predict(X_new)
    # Classificando a flor com base na previsão
    if prediction == 1:
        return "Outras"
    else:
        return "Iris-setosa"

# Loop iterativo para permitir que o usuário insira novos dados para fazer previsões
while True:
    # Solicitando os dados de entrada do usuário
    sepal_length = float(input("Digite o comprimento da sépala (cm): "))
    sepal_width = float(input("Digite a largura da sépala (cm): "))
    
    # Fazendo a previsão e exibindo o resultado
    prediction = predict_flower(sepal_length, sepal_width)
    print(f"A flor é classificada como: {prediction}")

    # Perguntando ao usuário se deseja continuar fazendo previsões
    continue_option = input("Deseja fazer outra previsão? (S/N): ").upper()
    if continue_option != 'S':
        break
