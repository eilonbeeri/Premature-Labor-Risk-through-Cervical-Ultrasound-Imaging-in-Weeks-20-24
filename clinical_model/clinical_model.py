import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score # <--- NEW IMPORTS
from sklearn.utils import class_weight

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading excel files...")
term_df = pd.read_excel("term birth clean data set.xlsx")
preterm_df = pd.read_excel("Pre term birth clean data set.xlsx")

# ---------------------------------------------------------
# 2. PREPARE DATA
# ---------------------------------------------------------
preterm_df['label'] = 1  # Target
term_df['label'] = 0     # Control

# Merge
df_full = pd.concat([preterm_df, term_df], ignore_index=True)

# Select only valid clinical features (No outcomes!)
features = [
    'Age', 'Pregnancy_Number', 'Num_Prev_births', 'Prev_preterm_birth', 
    'Prev_csections_count', 'Gastational_Age_exam_week_total', 
    'cervical_length_cm', 'IVF', 'smoking', 'Pre_Diabetes', 
    'Gestational_Diabetes', 'hypertension', 'Other_Conditions', 
    'conization', 'fetal_sex'
]

X = df_full[features].copy()
y = df_full['label'].copy()

# ---------------------------------------------------------
# 3. ROBUST CLEANING
# ---------------------------------------------------------
print("Cleaning data...")

# A. Binary Columns -> Fill NaN with 0
binary_cols = ['IVF', 'smoking', 'Pre_Diabetes', 'Gestational_Diabetes', 
               'hypertension', 'Other_Conditions', 
               'conization', 'Prev_preterm_birth']
for col in binary_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# B. Fetal Sex -> Map m/f to 1/0
X['fetal_sex'] = X['fetal_sex'].replace({'m': 1, 'f': 0, 'M': 1, 'F': 0})
X['fetal_sex'] = pd.to_numeric(X['fetal_sex'], errors='coerce').fillna(0)

# C. Numeric Columns -> Fill NaN with Median
numeric_cols = ['Age', 'Pregnancy_Number', 'Num_Prev_births', 
                'Prev_csections_count', 'cervical_length_cm', 
                'Gastational_Age_exam_week_total']
# Treat 0 weeks as missing
X['Gastational_Age_exam_week_total'] = X['Gastational_Age_exam_week_total'].replace(0, np.nan)

for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())

# ---------------------------------------------------------
# 4. SPLIT & SCALE
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. CALCULATE CLASS WEIGHTS
# ---------------------------------------------------------
print("Calculating class weights to fix imbalance...")

class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

weights_dict = {0: class_weights_vals[0], 1: class_weights_vals[1]}
print(f"Weights Applied: {weights_dict}")

# ---------------------------------------------------------
# 6. TRAIN MODEL
# ---------------------------------------------------------
print("Training model...")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', Recall(name='recall')])

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test), 
    epochs=100, 
    batch_size=32,
    class_weight=weights_dict, 
    verbose=0 
)

# ---------------------------------------------------------
# 7. EVALUATE & SAVE REPORT
# ---------------------------------------------------------
print("Generating reports...")

y_pred_prob = model.predict(X_test_scaled).ravel() # Get probabilities
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Calculate Metrics ---
auc_score = roc_auc_score(y_test, y_pred_prob)
ap_score = average_precision_score(y_test, y_pred_prob) # Average Precision

# Create report strings
report_str = classification_report(y_test, y_pred, target_names=['Term (Healthy)', 'Preterm (Early)'])
cm = confusion_matrix(y_test, y_pred)

print("--- NEW REPORT (Weighted) ---")
print(f"AUC Score: {auc_score:.4f}")
print(f"Avg Precision (AP): {ap_score:.4f}") # <--- Print to console
print(report_str)
print("Confusion Matrix:\n", cm)

# Save Text Report
with open("model_performance_report_weighted.txt", "w", encoding="utf-8") as f:
    f.write("--- DEEP LEARNING MODEL REPORT (WEIGHTED) ---\n")
    f.write(f"Class Weights Used: {weights_dict}\n\n")
    f.write(f"AUC SCORE: {auc_score:.4f}\n")
    f.write(f"AVG PRECISION (AP): {ap_score:.4f}\n\n") # <--- Added to File
    f.write("CONFUSION MATRIX:\n")
    f.write(f"{cm}\n\n")
    f.write("(Top-Left: True Negatives, Top-Right: False Positives)\n")
    f.write("(Bottom-Left: False Negatives, Bottom-Right: True Positives)\n\n")
    f.write("-" * 30 + "\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(report_str)

# Save Plots
# 1. Accuracy/Loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
plt.savefig('Accuracy_Loss_Plot_Weighted.png')

# 2. Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Term', 'Preterm'], 
            yticklabels=['Term', 'Preterm'])
plt.title('Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.savefig('Confusion_Matrix_Weighted.png')

# 3. ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('ROC_Curve_Weighted.png')

# --- NEW: 4. Precision-Recall Curve Plot ---
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AP = {ap_score:.2f})')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('Precision_Recall_Curve.png')

# Save Model
model.save('early_labor_model_weighted.keras')
joblib.dump(scaler, 'scaler_weighted.pkl')

print("\nSuccess! Files saved:")
print("1. model_performance_report_weighted.txt (Now includes AUC & AP)")
print("2. Accuracy_Loss_Plot_Weighted.png")
print("3. Confusion_Matrix_Weighted.png")
print("4. ROC_Curve_Weighted.png")
print("5. Precision_Recall_Curve.png (NEW)")
print("6. early_labor_model_weighted.keras")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 1. LOAD YOUR SAVED MODEL & DATA
# (Make sure these files are in the same folder)
print("Loading model and data...")
model = tf.keras.models.load_model('early_labor_model_weighted.keras') # Your best model
scaler = joblib.load('scaler_weighted.pkl')

# Load the test data again (to measure importance on)
term_df = pd.read_excel("term birth clean data set.xlsx")
preterm_df = pd.read_excel("Pre term birth clean data set.xlsx")

preterm_df['label'] = 1
term_df['label'] = 0
df_full = pd.concat([preterm_df, term_df], ignore_index=True)

# Select the EXACT SAME features you trained on (No tocolysis!)
features = [
    'Age', 'Pregnancy_Number', 'Num_Prev_births', 'Prev_preterm_birth', 
    'Prev_csections_count', 'Gastational_Age_exam_week_total', 
    'cervical_length_cm', 'IVF', 'smoking', 'Pre_Diabetes', 
    'Gestational_Diabetes', 'hypertension', 'Other_Conditions', 
    'conization', 'fetal_sex'
]

X = df_full[features].copy()
y = df_full['label'].copy()

# Fast Cleaning (Same as training script)
for col in ['IVF', 'smoking', 'Pre_Diabetes', 'Gestational_Diabetes', 'hypertension', 'Other_Conditions', 'conization', 'Prev_preterm_birth']:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
X['fetal_sex'] = X['fetal_sex'].replace({'m': 1, 'f': 0, 'M': 1, 'F': 0})
X['fetal_sex'] = pd.to_numeric(X['fetal_sex'], errors='coerce').fillna(0)
for col in ['Age', 'Pregnancy_Number', 'Num_Prev_births', 'Prev_csections_count', 'cervical_length_cm', 'Gastational_Age_exam_week_total']:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())

# Scale the data using your SAVED scaler
X_scaled = scaler.transform(X)

# 2. DEFINE THE "PERMUTATION IMPORTANCE" FUNCTION
def get_feature_importance(model, X, y, feature_names):
    # Step A: Get the model's original accuracy
    y_pred_orig = (model.predict(X, verbose=0) > 0.5).astype(int)
    baseline_acc = accuracy_score(y, y_pred_orig)
    print(f"Baseline Model Accuracy: {baseline_acc:.4f}")
    
    importances = []
    
    # Step B: Loop through every feature
    for i in range(X.shape[1]):
        # Create a copy of the data
        X_permuted = X.copy()
        
        # Scramble (Shuffle) ONLY this column
        np.random.shuffle(X_permuted[:, i])
        
        # Predict again with the scrambled data
        y_pred_perm = (model.predict(X_permuted, verbose=0) > 0.5).astype(int)
        perm_acc = accuracy_score(y, y_pred_perm)
        
        # The "Importance" is how much accuracy DROPPED
        drop = baseline_acc - perm_acc
        importances.append(drop)
        print(f"Feature: {feature_names[i]:<30} | Drop in Acc: {drop:.4f}")
        
    return pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# 3. RUN IT
print("\nCalculating importance (this might take 10 seconds)...")
importance_df = get_feature_importance(model, X_scaled, y, features)

# Sort results
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# 4. PLOT AND SAVE
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.xlabel("Importance (Drop in Accuracy)")
plt.title("Which Features Does YOUR Model Use?")
plt.tight_layout()
plt.savefig('my_model_feature_importance.png', dpi=300)
plt.show()

print("\nSaved plot as 'my_model_feature_importance.png'")
print(importance_df.sort_values(by='Importance', ascending=False))
