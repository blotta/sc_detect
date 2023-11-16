from datetime import datetime
from glob import glob
import time
from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from PIL import Image

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from scipy import stats

# datagen = ImageDataGenerator()

# train_dir = os.getcwd() + "/data/reorganized/"

# train_data_keras = datagen.flow_from_directory(directory=train_dir,
#                                                class_mode='categorical',
#                                                batch_size=16,
#                                                target_size=(32, 32))

# x, y = next(train_data_keras)
# for i in range(0, 15):
#     image = x[i].astype(int)
#     plt.imshow(image)
#     plt.show()

skin_df = pd.read_csv("data/HAM10000_metadata.csv")
print(skin_df['dx'].value_counts())

SIZE = 32

le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
print(list(le.classes_))

skin_df['label'] = le.transform(skin_df['dx'])
print(skin_df.sample(10))

# normalizacao
pkl_filename = 'skin_df_balanced.pkl'
if not os.path.exists(pkl_filename):
    print('generating new pickle file')
    df_0 = skin_df[skin_df['label'] == 0]
    df_1 = skin_df[skin_df['label'] == 1]
    df_2 = skin_df[skin_df['label'] == 2]
    df_3 = skin_df[skin_df['label'] == 3]
    df_4 = skin_df[skin_df['label'] == 4]
    df_5 = skin_df[skin_df['label'] == 5]
    df_6 = skin_df[skin_df['label'] == 6]

    n_samples = 500
    df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
    df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
    df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
    df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
    df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
    df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
    df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

    skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                                df_2_balanced, df_3_balanced,
                                df_4_balanced, df_5_balanced, df_6_balanced])

    print(skin_df_balanced['label'].value_counts())

    # Adiciona path de cada imagem no df

    image_path = {os.path.splitext(os.path.basename(x))[0]: x
                for x in glob(os.path.join('data/all_images/', '*.jpg'))}
    print(len(image_path))

    skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)

    skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))

    skin_df_balanced.to_pickle(pkl_filename)
else:
    print('using found pickle')
    skin_df_balanced = pd.read_pickle(pkl_filename)


print(skin_df_balanced)

X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255.
Y = skin_df_balanced['label']
Y_cat = to_categorical(Y, num_classes = 7)

x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

# Define the model

num_classes = 7
model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax')) # each of these 7 is a probability and the highest is our class
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])


# Train
batch_size = 16
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size= batch_size,
    validation_data=(x_test, y_test),
    verbose=2
)


# Accuracy
score = model.evaluate(x_test, y_test)

loss = score[0]
accuracy = score[1]
print(f'Accuracy: {accuracy * 100:.2f}')

# Model Evaluation
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# precision
# precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
# print(f'Precision: {precision * 100:.2f}')

# saving
model_filename = f'model_acc{accuracy*10000:.0f}_bs{batch_size}_e{epochs}_{time.strftime("%Y%m%d%H%M%S")}.h5'

model.save(model_filename)