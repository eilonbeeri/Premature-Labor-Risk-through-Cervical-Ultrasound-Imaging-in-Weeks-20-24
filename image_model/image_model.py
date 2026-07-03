import os, glob
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import tensorflow as tf

PRE_DIR  = "ROI PreTerm birth Ultrasound"   # label = 1
TERM_DIR = "ROI Term birth Ultrasound"  # label=0

IMG_SIZE = (224, 224)
BATCH = 8 # how many image processes before update weights
SEED = 42
exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

pre_paths = []
term_paths = []

for ext in exts:
    pre_paths += glob.glob(os.path.join(PRE_DIR, ext))
    term_paths += glob.glob(os.path.join(TERM_DIR, ext))

pre_paths = sorted(pre_paths)
term_paths = sorted(term_paths)

paths  = np.array(pre_paths + term_paths)
labels = np.array([1]*len(pre_paths) + [0]*len(term_paths))

# ====== split: train/val/test ======
# 1) split out test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    paths, labels, test_size=0.15, random_state=SEED, stratify=labels
)

# 2) split remaining into train (70%) and val (15%)
#    remaining is 85% -> val should be 15/85 of that remainder
val_frac_of_temp = 0.25 / 0.80

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_frac_of_temp, random_state=SEED, stratify=y_temp
)


def read_and_resize(path):
    # read grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = cv2.resize(img, IMG_SIZE)                 # resize whole image
    img = img.astype(np.float32) / 255.0            # normalize to 0..1
    img = np.repeat(img[..., None], 3, axis=-1)     # (H,W) -> (H,W,3)
    return img


print("Train/Val/Test:", X_train.shape[0], X_val.shape[0], X_test.shape[0])
print("Train counts [term,preterm]:", np.bincount(y_train.astype(int)))
print("Val   counts [term,preterm]:", np.bincount(y_val.astype(int)))
print("Test  counts [term,preterm]:", np.bincount(y_test.astype(int)))

#give more weight for the PreTerm
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()

class_weight = {
    0: (n0 + n1) / (2.0 * n0),
    1: (n0 + n1) / (2.0 * n1),
}

#turns paths to numric Image pixels values
def make_X_from_paths(path_list):
    return np.stack([read_and_resize(p) for p in path_list], axis=0)

X_train_img = make_X_from_paths(X_train)
X_val_img   = make_X_from_paths(X_val)
X_test_img  = make_X_from_paths(X_test)

y_train_f = y_train.astype(np.float32)
y_val_f   = y_val.astype(np.float32)
y_test_f  = y_test.astype(np.float32)

#Modle EfficientNetB0
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        "accuracy"
    ]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
]

history = model.fit(
    X_train_img, y_train_f,
    validation_data=(X_val_img, y_val_f),
    epochs=30,
    batch_size=BATCH,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

val_true = y_val_f.astype(int)
val_prob = model.predict(X_val_img).ravel()

thresholds = np.round(np.arange(0.4, 0.5, 0.01), 2)

best_threshold = None
best_specificity = -1
best_precision = -1
best_f1 = -1

max_allowed_missed_preterm = 2

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    cm = skm.confusion_matrix(val_true, val_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall_preterm = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_preterm = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_preterm = skm.f1_score(val_true, val_pred, pos_label=1)

    print("\nVALIDATION threshold:", threshold)
    print(cm)
    print("Missed Preterm:", fn)
    print("Preterm recall:", round(recall_preterm, 3))
    print("Preterm precision:", round(precision_preterm, 3))
    print("Specificity:", round(specificity, 3))
    print("Preterm F1:", round(f1_preterm, 3))

    # Choose threshold with FN <= 1 and highest specificity
    # If specificity is tied, choose better precision/F1
    if fn <= max_allowed_missed_preterm:
        if (
            specificity > best_specificity or
            (specificity == best_specificity and precision_preterm > best_precision) or
            (specificity == best_specificity and precision_preterm == best_precision and f1_preterm > best_f1)
        ):
            best_specificity = specificity
            best_precision = precision_preterm
            best_f1 = f1_preterm
            best_threshold = threshold

# Fallback: if no threshold satisfies FN <= 1
if best_threshold is None:
    print("\nNo threshold in this range had FN <= 1.")
    print("Choosing threshold with best Preterm F1 instead.")

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        val_pred = (val_prob >= threshold).astype(int)
        f1_preterm = skm.f1_score(val_true, val_pred, pos_label=1)

        if f1_preterm > best_f1:
            best_f1 = f1_preterm
            best_threshold = threshold

print("\nChosen threshold from validation:", best_threshold)
print("Best validation specificity:", round(best_specificity, 3))
print("Best validation precision:", round(best_precision, 3))
print("Best validation F1:", round(best_f1, 3))

# =====================================================
# 2) Final evaluation on TEST set
# =====================================================

y_true = y_test_f.astype(int)
y_prob = model.predict(X_test_img).ravel()
y_pred = (y_prob >= best_threshold).astype(int)

print("\n=== FINAL TEST RESULTS ===")
print("Chosen threshold:", best_threshold)
print("TEST ROC-AUC:", round(skm.roc_auc_score(y_true, y_prob), 3))
print("TEST PR-AUC :", round(skm.average_precision_score(y_true, y_prob), 3))

print("Confusion matrix:")
print(skm.confusion_matrix(y_true, y_pred, labels=[0, 1]))

print(skm.classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Term", "Preterm"],
    digits=3
))

print("TEST ROC-AUC:", skm.roc_auc_score(y_test_f, y_prob).round(3))
print("TEST PR-AUC :", skm.average_precision_score(y_test_f, y_prob).round(3))
print("Confusion matrix:\n", skm.confusion_matrix(y_test_f.astype(int), y_pred))
print(skm.classification_report(y_test_f.astype(int), y_pred, digits=3))

# =====================================================
# Grad-CAM for ALL images in one folder
# =====================================================

import os, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Folder containing the images you want Grad-CAM for
# Change this to the folder you want
GRADCAM_INPUT_DIR = "ROI PreTerm birth Ultrasound"
# GRADCAM_INPUT_DIR = "ROI Term birth Ultrasound"

# Folder where Grad-CAM outputs will be saved
SAVE_DIR = "plots_ROI/gradcam_all_images"
os.makedirs(SAVE_DIR, exist_ok=True)

THRESH = 0.44   # use your chosen threshold

exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP")

gradcam_paths = []
for ext in exts:
    gradcam_paths += glob.glob(os.path.join(GRADCAM_INPUT_DIR, ext))

gradcam_paths = sorted(gradcam_paths)

print("Number of images found:", len(gradcam_paths))
print("Saving Grad-CAMs to:", os.path.abspath(SAVE_DIR))


# =====================================================
# Locate EfficientNet and last conv layer
# =====================================================

base = model.get_layer("efficientnetb0")

def find_last_conv_like(m):
    for layer in reversed(m.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No Conv2D/DepthwiseConv2D layer found")

last_conv_name = find_last_conv_like(base)
print("Using last conv-like layer:", last_conv_name)

conv_model = tf.keras.Model(
    inputs=base.input,
    outputs=base.get_layer(last_conv_name).output
)

gap_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][-1]
drop_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)][-1]
dense_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][-1]


# =====================================================
# Grad-CAM function
# =====================================================

def gradcam_heatmap(img_01):
    """
    img_01: image with shape (224, 224, 3), values in [0,1]
    returns heatmap with shape (224, 224)
    """

    img_tensor = tf.convert_to_tensor(img_01[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = tf.keras.applications.efficientnet.preprocess_input(img_tensor * 255.0)

        conv_out = conv_model(x)
        tape.watch(conv_out)

        z = gap_layer(conv_out)
        z = drop_layer(z, training=False)
        pred = dense_layer(z)

        # Grad-CAM for Preterm probability
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_out)

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)

    cam = cam[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    cam = tf.image.resize(cam[..., None], (img_01.shape[0], img_01.shape[1]))[..., 0]

    return cam.numpy()


# =====================================================
# Run Grad-CAM on all images in the folder
# =====================================================

results = []

for path in gradcam_paths:
    img = read_and_resize(path)

    prob = model.predict(img[None, ...], verbose=0).ravel()[0]
    pred_label = int(prob >= THRESH)

    heat = gradcam_heatmap(img)

    base_name = os.path.splitext(os.path.basename(path))[0]

    fig = plt.figure(figsize=(5.5, 4.5))
    plt.imshow(img[..., 0], cmap="gray")
    plt.imshow(heat, cmap="jet", alpha=0.30)
    plt.axis("off")
    plt.title(f"{base_name}\np_preterm={prob:.3f}, pred={pred_label}, threshold={THRESH}")

    out_path = os.path.join(SAVE_DIR, f"gradcam_{base_name}_p{prob:.3f}_pred{pred_label}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    results.append({
        "image_name": os.path.basename(path),
        "image_path": path,
        "p_preterm": prob,
        "pred_label": pred_label,
        "gradcam_path": out_path
    })

    print("Saved:", out_path)


# Save probabilities summary
results_df = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "gradcam_results_summary.csv")
results_df.to_csv(csv_path, index=False)

print("\nDone.")
print("Saved all Grad-CAM images to:", os.path.abspath(SAVE_DIR))
print("Saved summary CSV to:", os.path.abspath(csv_path))

import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
SAVE_DIR = "plots_ROI5"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_and_show(fig, filename):
    """Save to SAVE_DIR and also to current folder, then show."""
    p1 = os.path.join(SAVE_DIR, filename)
    p2 = filename
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", p1, "and", p2)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Term (0)", "Preterm (1)"])

fig, ax = plt.subplots(figsize=(5,4))
disp.plot(values_format="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (Test) - Weighted")
plt.tight_layout()
save_and_show(fig, "confusion_matrix_test_weighted.png")

# ======================================================
# 2) Training curves: Accuracy + Loss (from history)
# ======================================================
H = history.history

# Loss
fig = plt.figure(figsize=(6,4))
plt.plot(H["loss"], label="train")
plt.plot(H["val_loss"], label="val")
plt.title("Loss (Weighted)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_and_show(fig, "loss_curve_weighted.png")

# Accuracy
fig = plt.figure(figsize=(6,4))
plt.plot(H["accuracy"], label="train")
plt.plot(H["val_accuracy"], label="val")
plt.title("Accuracy (Weighted)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_and_show(fig, "accuracy_curve_weighted.png")

# ======================================================
# 3) ROC curve (TEST)
# ======================================================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc_val = auc(fpr, tpr)

fig = plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_val:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curve (Test) - Weighted")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_and_show(fig, "roc_curve_test_weighted.png")

# ======================================================
# 4) PR curve (TEST)
# ======================================================
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

fig = plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
plt.title("Precision-Recall Curve (Test) - Weighted")
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision (PPV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_and_show(fig, "pr_curve_test_weighted.png")

print("\nDone. All PNGs saved in:", SAVE_DIR)

