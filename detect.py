import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

input_path = 'input/image.jpg'
output_path = Path('output')
output_path.mkdir(parents=True, exist_ok=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = cv2.imread(input_path)
assert img is not None, f'Imagem não encontrada: {input_path}'

results = model(img)

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.imshow(results.render()[0])
plt.axis('off')
plt.title('Detecções da YOLOv5')
plt.show()

output_img_path = output_path / 'resultado.jpg'
cv2.imwrite(str(output_img_path), results.render()[0])

print(f'Imagem salva em: {output_img_path}')
