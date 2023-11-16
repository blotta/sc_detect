import cv2
import numpy as np

print(cv2.__version__)

# Carregar a imagem
imagem = cv2.imread('m1.png')

# Converter a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar um desfoque para suavizar a imagem e reduzir o ruído
imagem_desfocada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

# Detecção de bordas usando Canny
bordas = cv2.Canny(imagem_desfocada, 50, 150)

# Encontrar contornos na imagem
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhar os contornos na imagem original
cv2.drawContours(imagem, contornos, -1, (0, 255, 0), 2)

# Exibir a imagem resultante
cv2.imshow('Detecção de Contornos', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
