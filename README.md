# Projeto de criação de uma base de dados e treinamento da rede YOLO

Vamos criar um código em Python que utilize a biblioteca **YOLO** para detectar e classificar objetos em uma imagem **.jpg**, ler essa imagem a partir da pasta **input** e salvar uma nova imagem com as caixas delimitadoras e classificações na pasta **output**.

Para este exemplo, vamos utilizar a implementação do **YOLOv5**, que é amplamente usada e fácil de configurar com a biblioteca **ultralytics**.

## Passos:

1. Instalar as dependências necessárias.

2. Configurar a estrutura do projeto.

3. Escrever o código para detectar e classificar objetos.

## 1. Instalar as dependências necessárias

Instale as bibliotecas necessárias:

```python
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install git+https://github.com/ultralytics/yolov5.git
```

## 2. Configurar a estrutura do projeto

Estruture seu projeto da seguinte forma:

```
yolo_project/
├── input/
│   └── image.jpg
├── output/
└── detect.py
```

## 3. Escrever o código para detectar e classificar objetos

Crie um arquivo **detect.py** com o seguinte conteúdo:

```python
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Caminhos das pastas de entrada e saída
input_path = 'input/sua_imagem.jpg'  # Substitua pelo nome da sua imagem
output_path = Path('output')
output_path.mkdir(parents=True, exist_ok=True)

# Carregar o modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ler a imagem
img = cv2.imread(input_path)
assert img is not None, f'Imagem não encontrada: {input_path}'

# Detectar objetos na imagem
results = model(img)

# Plotar resultados
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.imshow(results.render()[0])
plt.axis('off')
plt.title('Detecções da YOLOv5')
plt.show()

# Salvar a imagem com caixas delimitadoras
output_img_path = output_path / 'resultado.jpg'
cv2.imwrite(str(output_img_path), results.render()[0])

print(f'Imagem salva em: {output_img_path}')
```

## Explicação do Código:

### 1. Carregar e preparar as bibliotecas:

Importamos as bibliotecas necessárias, incluindo torch, cv2, Path, e matplotlib.

### 2. Definir caminhos para os arquivos de entrada e saída:

Definimos os caminhos para a imagem de entrada e a pasta de saída. Criamos a pasta de saída se ela não existir.

### 3. Carregar o modelo YOLOv5 pré-treinado:

Utilizamos torch.hub.load para carregar o modelo YOLOv5 pré-treinado.

### 3. Ler e processar a imagem:

Lemos a imagem da pasta input e utilizamos o modelo YOLOv5 para detectar objetos na imagem.

### 4. Plotar e salvar a imagem com as detecções:

Utilizamos matplotlib para exibir a imagem com as detecções e cv2.imwrite para salvar a imagem na pasta output.

## Executando o Projeto

1. Coloque a imagem que você deseja processar na pasta input com o nome sua_imagem.jpg.

![image]()

2. Execute o script:

```
python detect.py
```

3. A imagem com as caixas delimitadoras e classificações será salva na pasta **output** com o nome **resultado.jpg**.

![resultado]()

## Contribuição <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="25" height="25" />

Sinta-se à vontade para contribuir com este projeto. Você pode abrir issues para relatar problemas ou fazer pull requests para melhorias.
