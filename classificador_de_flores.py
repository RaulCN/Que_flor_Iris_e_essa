import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Função para prever a classe da flor
def predict_flower():
    # Obter os valores de entrada da interface gráfica
    sepal_length = float(sepal_length_entry.get())
    sepal_width = float(sepal_width_entry.get())
    petal_length = float(petal_length_entry.get())
    petal_width = float(petal_width_entry.get())
    
    # Prever a classe da flor com base nos valores de entrada
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Atualizar a etiqueta de saída com a classe prevista
    output_label.config(text=f'Classe prevista: {iris.target_names[prediction[0]]}')

# Criar a janela principal
root = tk.Tk()
root.title('Classificador de Flores Iris')

# Texto informativo sobre o conjunto de dados Iris e características das flores
info_text = """
O conjunto de dados Iris é um dos conjuntos de dados mais populares em aprendizado de máquina.
Ele contém informações sobre 150 amostras de flores Iris, cada uma com 4 características: 
- Comprimento da sépala (cm)
- Largura da sépala (cm)
- Comprimento da pétala (cm)
- Largura da pétala (cm)

As flores são classificadas em 3 espécies: Setosa, Versicolor e Virginica.

Por favor, insira os valores das características acima para prever a classe da flor.
"""

info_label = ttk.Label(root, text=info_text, wraplength=400, justify='left')
info_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

# Criar e posicionar os widgets na janela
ttk.Label(root, text='Comprimento da Sépala (cm)').grid(row=1, column=0, padx=10, pady=5)
sepal_length_entry = ttk.Entry(root)
sepal_length_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text='Largura da Sépala (cm)').grid(row=2, column=0, padx=10, pady=5)
sepal_width_entry = ttk.Entry(root)
sepal_width_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(root, text='Comprimento da Pétala (cm)').grid(row=3, column=0, padx=10, pady=5)
petal_length_entry = ttk.Entry(root)
petal_length_entry.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(root, text='Largura da Pétala (cm)').grid(row=4, column=0, padx=10, pady=5)
petal_width_entry = ttk.Entry(root)
petal_width_entry.grid(row=4, column=1, padx=10, pady=5)

predict_button = ttk.Button(root, text='Prever Classe', command=predict_flower)
predict_button.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

output_label = ttk.Label(root, text='')
output_label.grid(row=6, column=0, columnspan=2, padx=10, pady=5)

# Iniciar a interface gráfica
root.mainloop()
