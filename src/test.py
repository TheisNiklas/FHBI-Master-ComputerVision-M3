import tensorflow as tf
from utils.modelLoader import ModelLoader
import pandas as pd
from utils.utilities import buildRunName
import numpy as np


@tf.function
def decode_img(img_path):
    """
    function read image from filepath and format it into a tensor
    :param img_path: filepath of the image
    :return: decodes image as tensor
    """
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def process_path(file_path, labels):
    label = {'out_age_prediction': tf.reshape(tf.keras.backend.cast(labels[0], tf.keras.backend.floatx()), (1, 1)),
             'out_face_detection': tf.reshape(tf.keras.backend.cast(labels[1], tf.keras.backend.floatx()), (1, 1)),
             'out_mask_detection': tf.reshape(tf.keras.backend.cast(labels[2], tf.keras.backend.floatx()), (1, 1))}
    img = decode_img(file_path)
    return img, label

def group_ages(age: int):
    current_range = [
        ( 0, 0),
        ( 1,10),
        (11,20),
        (21,30),
        (31,40),
        (41,50),
        (51,60),
        (61,70),
        (71,80),
        (81,90),
        (91,100)
    ]
    if isinstance(age, int) and age >= current_range[0] and age <= current_range[1]:
        return current_range.index(current_range)
    else:
        return age

def create_dataset(data):
    data = tf.data.Dataset.from_tensor_slices(
        (data["Filepath"], data[["Face", "Mask", "Age"]]))
    ds = data.map(process_path)
    ds = ds.batch(32)
    return ds


metaData = pd.read_json("../data_meta/meta_all.json")
metaData_train = metaData.iloc[0:int(metaData.__len__() * 0.7)]
metaData_val =  metaData.iloc[int(metaData.__len__() * 0.7)+1:]

train_ds = create_dataset(metaData_train)
val_ds = create_dataset(metaData_val)

model = ModelLoader().loadMobileNetV1Multi(10)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={
        "out_age_prediction": tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=-1
            ),
        "out_face_detection": tf.keras.losses.BinaryCrossentropy(),
        "out_mask_detection": tf.keras.losses.BinaryCrossentropy(),
    },
    loss_weights=[1/3, 1/3, 1/3],
    metrics={
        "out_age_prediction": [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
        "out_face_detection": tf.keras.metrics.Accuracy(),
        "out_mask_detection": tf.keras.metrics.Accuracy(),
    },
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)    