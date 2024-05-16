# Projeto Machine Learning

"""
1 - Importar as bibliotecas necessárias
2 - Criar um conjunto de dados ficticios com o diâmetro e o preço de algumas pizzas.
3 - Visualizar os dados em um gráfico de dispersão
4 - Criar e treinar um modelo de regressão linear com os dados
5 - Avaliar o desempenho do modelo com algumas métricas
6 - Fazer uma previsão para uma nova pizza
"""

# Importando as Bibliotecas
import numpy as np # Manipular os Arrays
import matplotlib.pyplot as plt # Visualizar os dados
from sklearn.linear_model import LinearRegression # Para criar modelo de Regressão Linear
from sklearn.metrics import mean_squared_error, r2_score # Avaliar o modelo

# Criar o conjunto de dados com o diâmetro e o preço das pizzas
diametro_pizza = np.array([6, 8, 10, 14, 18]) # Diametro e Polegada Pizzas
valor_pizza = np.array([7, 9, 13, 17.5, 24.99]) # Preço das Pizzas

# Visualizar os dados em um grafico de dispersão
plt.scatter(diametro_pizza, valor_pizza) # Criar o Grafico
x = diametro_pizza # Diametro = x
y = valor_pizza # valor = y
plt.xlabel('Diametro em Polegadas') # Adicionar o rótulo do eixo x
plt.ylabel('Preço em R$') # Adicionar o rótulo do eixo y
plt.title('Preço da Pizza em razão de seu diâmetro') # Adicionar o titulo do Grafico
# plt.show() # Mostrar o Gráfico

# Criar e treinar um modelo de regressão linear com os dados
model = LinearRegression() # Instanciar o modelo
x = x.reshape(-1, 1) # Tranformar o array X em uma matriz de coluna
model.fit(x, y) # Treinar o modelo com os dados

# Avaliar o desempenho do modelo com algumas métricas
y_pred = model.predict(x) # Fazer as previsões para os dados de treino
mse = mean_squared_error(y, y_pred) # Calcular o erro quadratico médio
"""
Erro quadratico médio e um esquema utilizado na estatisticas, soma de todos 
os resultados tidos como 'erro' em relação a previsão inicial e, posteriomente,
dividi-los pela quantidade de valores somados.
"""
r2 = r2_score(y, y_pred) # Calcular o coeficiente de determinação
"""
Calculo de coeficiente de determinação Tambem chamado de R², é uma medida de ajuste de um modelo estatístico linear, como
a regressão linear simples ou múltipla, os valores observados de uma váriavel 
aleatória. O r² varia entre 0 e 1.
"""
print(f'{mse:.2f}') # Tranforma um n° grande em duas casas decimais sintaxe: f'{variavel:.2f}'
print(f'{r2:.2f}') # Imprimir o R² com duas casas decimais

# Fazer previsão para uma nova pizza
x_new = np.array([12]) # Diametro de uma nova pizza
y_new = model.predict(x_new.reshape(-1, 1)) # Fazer a previsão para uma nova pizza

print(f'Uma pizza de {x_new[0]} diâmetros vai custar: R$ {y_new[0]:.2f}')
# Vai mostrar o preço da Pizza conforme o diâmetro

