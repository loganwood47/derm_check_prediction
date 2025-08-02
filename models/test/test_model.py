import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from preprocess_test import get_processed_dataframe

test_df = pd.read_csv('ISIC_2019_Test_GroundTruth.csv')

MALIGNANT = ['MEL', 'BCC', 'AK', 'SCC']

test_df[MALIGNANT] = test_df[MALIGNANT].astype(int)

test_df['label'] = test_df[MALIGNANT].any(axis=1).astype(int)

test_df = test_df[['image', 'label']]

TEST_IMAGE_DIR = 'data/test/ISIC_2019_Test_Input'

X_test, y_test = get_processed_dataframe(TEST_IMAGE_DIR, test_df)

model = load_model('models/best_model.h5')

y_probs = model.predict(X_test)
y_preds = (y_probs > 0.5).astype(int).flatten()

if __name__ == '__main__':
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_preds))

    print("\nClassification Report:")
    print(classification_report(y_test, y_preds))

    print(f"\nROC AUC: {roc_auc_score(y_test, y_probs):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.show()