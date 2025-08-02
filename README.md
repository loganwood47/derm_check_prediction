# derm_check_prediction
POC Image recognition project using CNN's (specifically the [MobileNetV2](https://arxiv.org/abs/1801.04381) model) to predict whether a mole is benign or not. Not a diagnostic tool: intended to triage dermatology appointments

## Data
Data referenced in model training can be downloaded here: https://challenge.isic-archive.com/data/#2019

## Usage
Run `streamlit run app.py` to launch Streamlit classification app in browser!

### Training Models:
Preprocessing: upload training data and run `models/train/preprocess_train.py` to sort into correct benign/malignant directories (only need to do this once)

Run `models/train/train_model.py` to train model on sorted data. Outputs `best_model.h5` binary by default but rename as you please to preserve different models

### Testing Models:
Run `models/test/test_model.py` to test desired model and analyze accuracy