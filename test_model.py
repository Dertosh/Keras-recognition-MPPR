from keras.models import load_model
from keras.utils import plot_model

import cv2
import numpy as np

try:
    MODEL = load_model("digits_cls.ckpt")
except IOError:
    print("E: No digits_cls")
    exit()
MODEL.summary()
#print(MODEL.to_json())
#plot_model(MODEL, to_file='model.png', show_shapes=True)



img_path = 'examples/fish_bike.jpg'

try:
    im = cv2.imread("test.bmp")
except IOError:
    #server.send(request)
    print("E: No image")
    exit()
# Конвертаця в серый слой и применение фильтра Гаусса
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Поиск изображения
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Поиск контуров на изображении
# im_th - исходное изображение
# RETR_EXTERNAL - режим поиска контуров
# CHAIN_APPROX_SIMPLE - метод приближения контуров
_, ctrs, hier = cv2.findContours(
im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rect = cv2.boundingRect(sorted(ctrs, key=cv2.contourArea, reverse=True)[0])  # получить самый большой контур

# Вычленить полученный прямоугольник из изображения
leng = int(rect[3] * 1.6)
pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
# Подготовить MNIST-изображение
roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
roi = cv2.dilate(roi, (3, 3))

#len(recognition_image)
recognition_image = roi.reshape((1, -1))



out = MODEL.predict(recognition_image)
print('Распознаная цифра: ',np.argmax(out))

