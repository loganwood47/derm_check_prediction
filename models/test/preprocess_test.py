import pandas as pd
import os
import numpy as np
from PIL import Image


def preprocess_image(path, target_size=(224, 224)):
    img = Image.open(path).convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

def get_processed_dataframe(image_dir, test_df):
    X_test = []
    y_test = []
    processedCounter = 0

    for _, row in test_df.iterrows():
        img_path = os.path.join(image_dir, f"{row['image']}.jpg")
        if os.path.exists(img_path):
            X_test.append(preprocess_image(img_path))
            y_test.append(row['label'])
        processedCounter += 1
        print(f"Processed {processedCounter}/{len(test_df)} images: {row['image']} -> {row['label']}")

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_test, y_test

