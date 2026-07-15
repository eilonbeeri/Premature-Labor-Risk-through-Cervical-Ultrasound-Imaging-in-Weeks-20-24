import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from sklearn.utils import class_weight
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

# ==========================================
# 1. SETUP & PATHS
# ==========================================
BASE_DIR = "/content/drive/My Drive/final_project"

# Image Directories
PRE_IMG_DIR = os.path.join(BASE_DIR, "ROI PreTerm birth Ultrasound")
TERM_IMG_DIR = os.path.join(BASE_DIR, "ROI Term birth Ultrasound")

# Clinical Excel Files
PRE_CLINICAL_FILE = os.path.join(BASE_DIR, "clinical_model", "Pre term birth clean data set.xlsx")
TERM_CLINICAL_FILE = os.path.join(BASE_DIR, "clinical_model", "term birth clean data set.xlsx")

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH = 8
SEED = 42

ID_COLUMN_NAME = "patient_number"

CLINICAL_FEATURES = [
    'Age', 'Pregnancy_Number', 'Num_Prev_births', 'Prev_preterm_birth',
    'Prev_csections_count', 'Gastational_Age_exam_week_total',
    'cervical_length_cm', 'IVF', 'smoking', 'Pre_Diabetes',
    'Gestational_Diabetes', 'hypertension', 'Other_Conditions',
    'conization', 'fetal_sex'
]

# ==========================================
# 2. LOAD & ROBUSTLY CLEAN CLINICAL DATA
# ==========================================
print("Loading excel files...")
term_df = pd.read_excel(TERM_CLINICAL_FILE)
preterm_df = pd.read_excel(PRE_CLINICAL_FILE)

preterm_df['label'] = 1
term_df['label'] = 0

# Merge first to calculate medians globally (just like clinical_model.py)
df_full = pd.concat([preterm_df, term_df], ignore_index=True)

print("Cleaning clinical data...")
# A. Binary Columns -> Fill NaN with 0
binary_cols = ['IVF', 'smoking', 'Pre_Diabetes', 'Gestational_Diabetes',
               'hypertension', 'Other_Conditions',
               'conization', 'Prev_preterm_birth']
for col in binary_cols:
    df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0)

# B. Fetal Sex -> Map m/f to 1/0
df_full['fetal_sex'] = df_full['fetal_sex'].replace({'m': 1, 'f': 0, 'M': 1, 'F': 0})
df_full['fetal_sex'] = pd.to_numeric(df_full['fetal_sex'], errors='coerce').fillna(0)

# C. Numeric Columns -> Fill NaN with Median
numeric_cols = ['Age', 'Pregnancy_Number', 'Num_Prev_births',
                'Prev_csections_count', 'cervical_length_cm',
                'Gastational_Age_exam_week_total']
# Treat 0 weeks as missing
df_full['Gastational_Age_exam_week_total'] = df_full['Gastational_Age_exam_week_total'].replace(0, np.nan)

for col in numeric_cols:
    df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
    df_full[col] = df_full[col].fillna(df_full[col].median())

# ==========================================
# 3. ALIGN IMAGES WITH CLEANED DATA
# ==========================================
print("Aligning images with clinical records...")

def read_and_resize(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.repeat(img[..., None], 3, axis=-1)
    return img

aligned_img, aligned_clin, aligned_lab = [], [], []

for index, row in df_full.iterrows():
    # Safely get patient ID string
    patient_id = str(row[ID_COLUMN_NAME]).replace('.0', '')
    label = int(row['label'])

    # Check the appropriate folder based on the label
    img_dir = PRE_IMG_DIR if label == 1 else TERM_IMG_DIR
    search_pattern = os.path.join(img_dir, f"{patient_id}.*")
    found_files = glob.glob(search_pattern)

    if len(found_files) > 0:
        img_path = found_files[0]
        img = read_and_resize(img_path)

        if img is not None:
            aligned_img.append(img)
            # Extract ONLY the 15 explicitly specified clinical features
            clin_data = row[CLINICAL_FEATURES].values.astype(np.float32)
            aligned_clin.append(clin_data)
            aligned_lab.append(label)

X_img = np.array(aligned_img)
X_clin = np.array(aligned_clin)
y = np.array(aligned_lab).astype(np.float32)

print(f"Successfully aligned {len(y)} complete patient records.")
print(f"Preterm (1): {sum(y == 1)}, Term (0): {sum(y == 0)}")

# ==========================================
# 4. SPLIT, SCALE & WEIGHTS
# ==========================================
idx = np.arange(len(y))
# 1) Split out test (20% to match clinical_model.py)
idx_train_temp, idx_test, y_train_temp, y_test = train_test_split(
    idx, y, test_size=0.20, random_state=SEED, stratify=y
)

# 2) Split remaining into train (65%) and val (15% of total) for early stopping
val_frac = 0.15 / 0.80
idx_train, idx_val, y_train, y_val = train_test_split(
    idx_train_temp, y_train_temp, test_size=val_frac, random_state=SEED, stratify=y_train_temp
)

X_train_img, X_val_img, X_test_img = X_img[idx_train], X_img[idx_val], X_img[idx_test]
X_train_clin, X_val_clin, X_test_clin = X_clin[idx_train], X_clin[idx_val], X_clin[idx_test]

# Scale
scaler = StandardScaler()
X_train_clin = scaler.fit_transform(X_train_clin)
X_val_clin = scaler.transform(X_val_clin)
X_test_clin = scaler.transform(X_test_clin)

# Class Weights (using sklearn formula to match clinical_model.py)
print("Calculating class weights to fix imbalance...")
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
weights_dict = {0: class_weights_vals[0], 1: class_weights_vals[1]}
print(f"Weights Applied: {weights_dict}")

# ==========================================
# 5. BUILD MULTIMODAL ARCHITECTURE
# ==========================================
print("\nBuilding multimodal model...")

# --- Image Branch ---
img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_input")
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

x_img = tf.keras.applications.efficientnet.preprocess_input(img_input * 255.0)
x_img = base_model(x_img, training=False)
x_img = GlobalAveragePooling2D()(x_img)
x_img = Dropout(0.3)(x_img)
img_features = Dense(128, activation="relu")(x_img)

# --- Clinical Branch ---
clin_input = Input(shape=(len(CLINICAL_FEATURES),), name="clinical_input")
x_clin = Dense(64, activation="relu")(clin_input)
x_clin = Dropout(0.3)(x_clin)
x_clin = Dense(32, activation="relu")(x_clin)
clin_features = Dropout(0.2)(x_clin)

# --- Fusion ---
merged = Concatenate()([img_features, clin_features])
x_comb = Dense(128, activation="relu")(merged)
x_comb = Dropout(0.3)(x_comb)
x_comb = Dense(64, activation="relu")(x_comb)
outputs = Dense(1, activation="sigmoid", name="output")(x_comb)

model = Model(inputs=[img_input, clin_input], outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryFocalCrossentropy(
    gamma=2.0,  # Forces the model to focus strictly on the "hard" examples
    alpha=0.85  # Heavily weights the minority class (Preterm)
),
    metrics=['accuracy', tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.AUC(name="pr_auc", curve="PR")]
)

# ==========================================
# 6. TRAINING
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=4, min_lr=1e-6)
]

print("\nStarting training...")
history = model.fit(
    x={"image_input": X_train_img, "clinical_input": X_train_clin}, y=y_train,
    validation_data=({"image_input": X_val_img, "clinical_input": X_val_clin}, y_val),
    epochs=100, batch_size=BATCH, class_weight=weights_dict, callbacks=callbacks, verbose=1
)

# ==========================================
# 7. EVALUATE & SAVE REPORTS
# ==========================================
print("\nGenerating reports...")
y_pred_prob = model.predict({"image_input": X_test_img, "clinical_input": X_test_clin}).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = y_test.astype(int)

auc_score = roc_auc_score(y_true, y_pred_prob)
ap_score = average_precision_score(y_true, y_pred_prob)
report_str = classification_report(y_true, y_pred, target_names=['Term', 'Preterm'])
cm = confusion_matrix(y_true, y_pred)

print("--- MULTIMODAL MODEL REPORT ---")
print(f"AUC Score: {auc_score:.4f}")
print(f"Avg Precision (AP): {ap_score:.4f}")
print(report_str)

# Save Text Report
with open(os.path.join(BASE_DIR, "multimodal_performance_report.txt"), "w", encoding="utf-8") as f:
    f.write("--- MULTIMODAL DEEP LEARNING MODEL REPORT ---\n")
    f.write(f"Class Weights Used: {weights_dict}\n\n")
    f.write(f"AUC SCORE: {auc_score:.4f}\n")
    f.write(f"AVG PRECISION (AP): {ap_score:.4f}\n\n")
    f.write("CONFUSION MATRIX:\n")
    f.write(f"{cm}\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(report_str)

# 1. Accuracy/Loss Plot
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'Multimodal_Accuracy_Loss.png'))

# 2. Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Term', 'Preterm'], yticklabels=['Term', 'Preterm'])
plt.title('Multimodal Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.savefig(os.path.join(BASE_DIR, 'Multimodal_Confusion_Matrix.png'))

# 3. ROC Curve Plot
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multimodal ROC Curve')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, 'Multimodal_ROC_Curve.png'))

# 4. Precision-Recall Plot
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AP = {ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multimodal Precision-Recall Curve')
plt.legend(loc="lower left"); plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, 'Multimodal_PR_Curve.png'))

# Save Model & Scaler
model.save(os.path.join(BASE_DIR, 'multimodal_model.keras'))
joblib.dump(scaler, os.path.join(BASE_DIR, 'multimodal_scaler.pkl'))

print("\nSuccess! All models, reports, and plots have been saved to your Google Drive.")

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("\n--- GENERATING MULTIMODAL GRAD-CAM HEATMAPS ---")

SAVE_DIR = os.path.join(BASE_DIR, "plots", "gradcam")
os.makedirs(SAVE_DIR, exist_ok=True)

def save_and_show(fig, filename):
    p1 = os.path.join(SAVE_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", p1)

# ---------- Get Predictions ----------
y_true = y_test.astype(int)
y_prob = model.predict({"image_input": X_test_img, "clinical_input": X_test_clin}, verbose=0).ravel()
THRESH = 0.50
y_pred = (y_prob >= THRESH).astype(int)

# -------------------------------------------------------------
# FOOLPROOF LAYER EXTRACTION BY MATRIX SHAPE
# -------------------------------------------------------------
denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

def get_dense_by_shape(in_dim, out_dim):
    """Finds the exact Dense layer by looking at its mathematical kernel shape"""
    for l in denses:
        w = l.get_weights()
        if len(w) > 0 and w[0].shape == (in_dim, out_dim):
            return l
    raise ValueError(f"Missing Dense layer with shape ({in_dim}, {out_dim})")

# Grab all the layers using their exact mathematical dimensions
base_eff_model = model.get_layer("efficientnetb0")
gap_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][0]
concat_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate)][0]

img_dense = get_dense_by_shape(1280, 128)
clin_dense_1 = get_dense_by_shape(15, 64)
clin_dense_2 = get_dense_by_shape(64, 32)
comb_dense_1 = get_dense_by_shape(160, 128)  # 128 (img) + 32 (clin) = 160
comb_dense_2 = get_dense_by_shape(128, 64)
final_dense = get_dense_by_shape(64, 1)

def gradcam_heatmap_multimodal(img_01, clin_data):
    """Manually pushes the data through the layers to bypass Keras graph bugs."""
    img_tensor = tf.convert_to_tensor(img_01[None, ...], dtype=tf.float32)
    clin_tensor = tf.convert_to_tensor(clin_data[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        # 1. Image Branch
        x_img = tf.keras.applications.efficientnet.preprocess_input(img_tensor * 255.0)
        conv_out = base_eff_model(x_img, training=False)
        tape.watch(conv_out) # Watch the visual feature map!

        z_img = gap_layer(conv_out)
        img_feats = img_dense(z_img)

        # 2. Clinical Branch
        z_clin = clin_dense_1(clin_tensor)
        clin_feats = clin_dense_2(z_clin)

        # 3. Fusion & Head (No Dropouts needed for inference!)
        merged = concat_layer([img_feats, clin_feats])
        z_comb = comb_dense_1(merged)
        z_comb = comb_dense_2(z_comb)
        pred = final_dense(z_comb)

        loss = pred[:, 0]

    # Calculate gradients of the final prediction w.r.t the visual feature map
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Overlay the weights onto the feature map
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0]

    # Normalize to 0-1
    max_val = tf.reduce_max(cam)
    if max_val == 0:
        max_val = 1e-8
    cam = cam / max_val

    # Resize the heatmap to match the 224x224 ultrasound image
    cam = tf.image.resize(cam[..., None], (img_01.shape[0], img_01.shape[1]))[..., 0]
    return cam.numpy()

# ---------- Choose Examples ----------
tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

print(f"Available for Grad-CAM -> TP: {len(tp_idx)}, FP: {len(fp_idx)}, FN: {len(fn_idx)}")

def pick_confident(idxs, k=2, mode="high"):
    idxs = np.array(list(idxs))
    if len(idxs) == 0: return []
    scores = y_prob[idxs]
    order = np.argsort(scores)
    if mode == "high":
        idxs = idxs[order[::-1]]
    else:
        idxs = idxs[order]
    return idxs[:min(k, len(idxs))].tolist()

sel_tp = pick_confident(tp_idx, 2, mode="high")
sel_fp = pick_confident(fp_idx, 2, mode="high")
sel_fn = pick_confident(fn_idx, 2, mode="low")

selected = [("TP", i) for i in sel_tp] + [("FP", i) for i in sel_fp] + [("FN", i) for i in sel_fn]

# --------- Build Grid Figure ----------
if len(selected) > 0:
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.ravel()

    for j, (tag, idx) in enumerate(selected):
        if j >= len(axes): break
        img = X_test_img[idx]
        clin = X_test_clin[idx]

        # Generate the heatmap using BOTH inputs
        heat = gradcam_heatmap_multimodal(img, clin)

        axes[j].imshow(img[..., 0], cmap="gray")
        axes[j].imshow(heat, cmap="jet", alpha=0.35)
        axes[j].axis("off")
        axes[j].set_title(f"{tag} | True={y_true[idx]} Pred={y_pred[idx]} Prob={y_prob[idx]:.3f}")

    for k in range(j+1, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"Multimodal Grad-CAM Examples", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = f"gradcam_multimodal_grid.png"
    save_and_show(fig, fname)
else:
    print("Not enough positive predictions to generate a grid. Try lowering THRESH.")
