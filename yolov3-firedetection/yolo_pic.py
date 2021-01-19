from yolo3.model import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    img = input('Input imagename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error!')
        continue
    else:
        rel_image = yolo.detect_image(image)
        rel_image.show()
yolo.close_session()
