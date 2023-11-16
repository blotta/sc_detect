import os
import keras
import numpy as np
from PIL import Image

SIZE = 32

model = keras.models.load_model('model_acc7337_bs16_e50_20231116022632.h5')

class_codes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class_names = {
    'nv': 'Nevo melanocítico',
    'mel': 'Melanoma',
    'bkl': 'Lesões Benignas Semelhantes à Ceratose',
    'bcc': 'Carcinoma de Células Basais',
    'akiec': 'Queratoses Actínicas',
    'vasc': 'Lesões Vasculares',
    'df': 'Dermatofibroma'
}

def detect_skin_lesion(image_path):
    img = np.asarray(Image.open(image_path).resize((SIZE, SIZE)).convert("RGB"))
    img = img/255.
    img = np.expand_dims(img, axis=0)
    # print(img)

    prediction = model.predict(img)
    print(prediction)
    highest_perc_idx = np.argmax(prediction)
    highest_perc = prediction[0][highest_perc_idx]
    class_code = class_codes[highest_perc_idx]
    class_name = class_names[class_code]

    results = sorted(zip(class_codes, prediction[0]), key=lambda x: x[1], reverse=True)

    for code, perc in results:
        print(f"{perc * 100:2.2f}%: {class_names[code]} [{code}]")

    return results

test_img_dir = os.path.join(os.getcwd(), 'data', 'test_images/')
image_file = 'akiec02.png'

image_path = os.path.join(test_img_dir, image_file)


detect_skin_lesion(image_path)