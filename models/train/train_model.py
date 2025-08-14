import tensorflow as tf
import os
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import datetime
import shutil



def train_model(sample_fraction=1.0):
    img_size = (224, 224)
    batch_size = 32
    original_train_path = 'data/train'
    temp_train_path = 'data/train_sampled'

    #create sampled dataset
    if os.path.exists(temp_train_path):
        shutil.rmtree(temp_train_path)
    os.makedirs(temp_train_path)

    for class_name in os.listdir(original_train_path):
        class_dir = os.path.join(original_train_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        os.makedirs(os.path.join(temp_train_path, class_name))
        images = os.listdir(class_dir)
        random.shuffle(images)
        subset_size = max(1, int(len(images) * sample_fraction))
        selected = images[:subset_size]
        for img in selected:
            src = os.path.join(class_dir, img)
            dst = os.path.join(temp_train_path, class_name, img)
            shutil.copy(src, dst)

# create data generators
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        temp_train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        temp_train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # callbacks
    early_stop_cb = EarlyStopping(
        patience=3,
        restore_best_weights=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    checkpoint_cb = ModelCheckpoint(
        "best_model.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # build model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        verbose=2,
        callbacks=[checkpoint_cb, early_stop_cb, tensorboard_cb]
    )

    # save
    model.save(f'models/skin_cancer_model_{int(sample_fraction*100)}pct.h5')
    print(f'Model trained on {sample_fraction*100:.0f}% of training data saved.')

    return history

if __name__ == '__main__':
    for frac in [0.1, 0.25, 0.5, 0.75]:
    # for frac in [0.1]:
        print(f"\n=== Training on {frac*100:.0f}% of training data ===")
        train_model(sample_fraction=frac)
