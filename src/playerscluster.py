import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# x = Tempo de jogo semanal (em horas)
x = [43, 12, 55, 31, 59, 2, 37, 21, 25, 13, 15, 27, 48, 50, 36]
# y = Número de conquistas.
y = [66, 19, 82, 46, 88, 3, 51, 34, 38, 19, 27, 42, 72, 75, 54]

# Cria um array bidimensional com duas colunas, uma coluna x que representa o tempo jogado semanalmente em horas e uma coluna y que representa o nº de conquistas alcançadas
dados = np.array([[43, 66], [12, 19], [55, 82], [31, 46], [59, 88], [2, 3], [37, 51], [21, 34], [25, 38], [13, 19], [15, 27], [27, 42], [48, 72], [50, 75], [36, 54]])

# Cria um scaler pra normalizar os dados do array para que a média de cada variável seja 0 e o desvio padrão seja 1. 
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(dados)

# K-Means irá dividir os dados em 6 clusters e será ajustado aos dados normalizados.
kmeans = KMeans(n_clusters=6)
kmeans.fit(dados_normalizados)

# Atribui um rótulo a cada ponto, indicando a qual cluster ele pertence.
rotulos = kmeans.labels_

# Cria uma variável que contém os centroídes na escala normalizada e outra que contém os centroídes na escala original dos dados.
centroides_normalizados = kmeans.cluster_centers_
centroides = scaler.inverse_transform(kmeans.cluster_centers_)

#Cria um gráfico que mostra a distribuição de pontos em um plano bidimensional com a adição de marcadores para os centroides dos clusters
grafico_pontos = px.scatter(
  x=dados[:, 0],
  y=dados[:, 1],
  color=rotulos.astype(str),
  title="Cluster de Pontos",
  labels={"x": "tempo jogado semanalmente (h) ", "y": "nº de conquistas alcançadas "},
  color_discrete_sequence=px.colors.qualitative.Bold
)

grafico_centroides = go.Scatter(
  x=centroides[:, 0],
  y=centroides[:, 1],
  mode="markers",
  marker=dict(
    size=11,
    color='black',
    symbol='x'
  ),
  name="Centroides"
)

grafico_final = go.Figure(data=list(grafico_pontos.data) + [grafico_centroides])
grafico_final.show()
