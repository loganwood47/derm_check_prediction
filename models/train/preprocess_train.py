import os
import pandas as pd
from shutil import copyfile

# This just sorts the ISIC 2019 training dataset into benign and malignant folders
# for use in training a binary classifier

truth_csv_path = 'ISIC_2019_Training_GroundTruth.csv'
metadata_csv_path = 'ISIC_2019_Training_Metadata.csv'
image_dir = 'ISIC_2019_Training_Input'
output_dir = 'data/train'


os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'malignant'), exist_ok=True)


df = pd.read_csv(truth_csv_path)
dfMeta = pd.read_csv(metadata_csv_path)

MALIGNANT = ['MEL', 'BCC', 'AK', 'SCC'] #melanoma, basal cell carcinoma, actinic keratosis, squamous cell carcinoma

# Binary label: 1 = malignant, 0 = benign (assume anything not malignant is benign for simplicity)
df['label'] = df[MALIGNANT].any(axis=1).astype(int)
# TODO: sort these to get actual classification by type instead of binary

procCount = 0

for _, row in df.iterrows():
    img_id = row['image']
    label = 'malignant' if row['label'] == 1 else 'benign'
    src = os.path.join(image_dir, f'{img_id}.jpg')
    dst = os.path.join(output_dir, label, f'{img_id}.jpg')
    if os.path.exists(src):
        copyfile(src, dst)
    procCount += 1
    print(f'Processed {procCount}/{len(df)} images: {img_id} -> {label}')

if __name__ == '__main__':
    print(f'Processed {len(df)} images into {output_dir}')
    print(f'Benign images: {len(os.listdir(os.path.join(output_dir, "benign")))}')
    print(f'Malignant images: {len(os.listdir(os.path.join(output_dir, "malignant")))}')
    print('Preprocessing complete.')