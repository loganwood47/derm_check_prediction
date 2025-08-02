import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import datetime


img_size = (224, 224)
batch_size = 32

# Directory containing training images
train_path = 'data/train'

# preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

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

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=preds)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=2,
    callbacks=[checkpoint_cb, early_stop_cb, tensorboard_cb]  # open http://localhost:6006 for progress
)

if __name__ == '__main__':
    model.save('skin_cancer_model.h5')
    print('Model trained and saved as skin_cancer_model.h5')
