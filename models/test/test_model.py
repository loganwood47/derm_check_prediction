import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from preprocess_test import get_processed_dataframe


MALIGNANT = ['MEL', 'BCC', 'AK', 'SCC']
TEST_IMAGE_DIR = 'data/test/ISIC_2019_Test_Input'
MODEL_CONFIGS = [
    (2533, "models/skin_cancer_model_10pct.h5"),
    (6333, "models/skin_cancer_model_25pct.h5"),
    (12666, "models/skin_cancer_model_50pct.h5"),
    (18998, "models/skin_cancer_model_75pct.h5"),
    (25331, "models/best_model.h5")  # trained on all data
]
TARGET_FNR = 0.05  # Target false negative rate


# Load test set
test_df = pd.read_csv('ISIC_2019_Test_GroundTruth.csv')
test_df[MALIGNANT] = test_df[MALIGNANT].astype(int)
test_df['label'] = test_df[MALIGNANT].any(axis=1).astype(int)
test_df = test_df[['image', 'label']]

X_test, y_test = get_processed_dataframe(TEST_IMAGE_DIR, test_df)


def evaluate_model(model_path):
    model = load_model(model_path)
    y_probs = model.predict(X_test)
    y_preds = (y_probs > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 

    print(f"\nModel: {model_path}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_preds))
    print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.4f}")
    print(f"False Negative Rate: {fnr:.4f}")

    return fnr


# Fit an exponential decay curve: FNR = a * exp(-b * N) + c
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Predict number of samples needed for TARGET_FNR
def samples_for_fnr(target_fnr):
    # Solve target_fnr = a * exp(-b * N) + c
    return (np.log((target_fnr - c) / a) * -1 / b * 1000) if target_fnr > c else np.inf

if __name__ == '__main__':
    sample_sizes = []
    fnr_values = []

    for samples, model_path in MODEL_CONFIGS:
        print(f"\nEvaluating model trained on {samples} samples: {model_path}")
        fnr = evaluate_model(model_path)
        sample_sizes.append(samples)
        fnr_values.append(fnr)

    x_scaled = np.array(sample_sizes) / 1000 

    # Fit an exponential decay curve to the FNR values
    popt, _ = curve_fit(
        exp_decay, 
        sample_sizes, 
        fnr_values, 
        # p0=[0.3, 0.1, 0.05],
        bounds=([0, 0, 0], [np.inf, np.inf, 0.05])
    )

    a, b, c = popt
    print(f"\nFit parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    
    # Predict number of samples needed for TARGET_FNR
    needed_samples = samples_for_fnr(TARGET_FNR)
    print(f"Estimated samples needed for {TARGET_FNR*100:.1f}% FNR: {needed_samples:.1f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(sample_sizes, fnr_values, color='red', label='Observed FNR')
    x_line = np.linspace(min(sample_sizes), max(sample_sizes) * 1.5, 100)
    plt.plot(x_line, exp_decay(x_line, *popt), label='Fitted Curve')
    plt.axhline(TARGET_FNR, color='gray', linestyle='--', label=f'Target {TARGET_FNR*100:.1f}% FNR')
    plt.axvline(needed_samples, color='blue', linestyle='--', label=f'Needed ~{needed_samples:.0f} samples')
    plt.xlabel('Number of Training Samples (thousands)')
    plt.ylabel('False Negative Rate')
    plt.title('False Negative Rate vs Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.show()