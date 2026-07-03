# Premature Labor Risk through Cervical Ultrasound Imaging in Weeks 20-24
## Overview
Preterm birth (PTB), defined as delivery before 37 weeks of gestation, is a global challenge associated with substantial neonatal mortality and long-term morbidity. The standard clinical screening method relies on manual, operator-dependent measurements of cervical length via transvaginal ultrasound (TVUS). However, simple geometric metrics frequently fail to capture the subtle microscopic tissue-level remodeling that precedes spontaneous delivery.

This project introduces a software-based machine learning prototype designed for the **20–24 gestational week screening window**. The core architecture operates via a parallel dual-stream structure: a clinical branch capturing macro-level patient risk indicators and a computer vision branch capturing micro-textural signatures from ultrasound images. By utilizing an **Early Fusion (feature-level concatenation)** approach, the system learns multi-scale dependencies between historical risk patterns and localized anatomical markers to generate a highly sensitive, operator-independent risk probability assessment.

---

## System Architecture

The pipeline consists of three core components integrated into a cohesive data-processing stream:

### 1. Geometric Image Preprocessing (MATLAB)
To decouple model performance from external scanner configurations and manual measuring artifacts, raw TVUS scans undergo geometric normalization using a semi-automated script:
* **Resolution Upscaling & Calibration:** Images are scaled by a factor of 4. The operator draws a baseline over a known distance banner to establish a dynamic pixel-to-centimeter conversion factor.
* **Horizontal Tissue Alignment:** The operator marks a 4-point region of interest (ROI) around the cervical anatomy. The script calculates the structural angle ($\theta$), applies an active rotation matrix to align the tissue horizontally, and crops a normalized region strictly representing 2 cm of vertical tissue depth.

### 2. Dual-Stream Feature Learning (Python/TensorFlow)
* **Tabular Clinical Stream:** Processes 15 medical features (e.g., Age, Gravidity, Cervical Length, History of Preterm Birth). Continuous values are scaled using zero-mean, unit-variance standardization (`StandardScaler`), while missing parameters are handled robustly via median imputation to prevent scale bias. The representations are parsed through a Multi-Layer Perceptron (MLP) branch containing hidden layers of 64 and 32 neurons with specialized Dropout rates (0.3/0.2) for structural regularization.
* **Deep Image Stream:** Normalized ROI crops are resized to $224 \times 224 \times 3$ formats. The tensors pass through a frozen `EfficientNetB0` backbone pre-trained on ImageNet to extract abstract deep texture mappings via a `GlobalAveragePooling2D` layer and a 0.3 Dropout configuration.

### 3. Early Fusion Architecture
The network branches are merged at the feature level via a `Concatenate` layer, bridging the 128-neuron output of the image stream with the dense clinical embeddings. The fused tensor maps complex non-linear combinations across secondary shared layers (128 and 64 neurons) before projecting a final risk probability through a single-neuron sigmoid layer.

---

## Performance Summary

The integrated multimodal approach successfully met and exceeded all quantitative target thresholds defined during project planning, outperforming both single-modality baselines:

| Model Configuration | Target Accuracy Threshold | Achieved Test Accuracy | Achieved ROC-AUC | Achieved PR-AUC (AP) |
| :--- | :--- | :--- | :--- | :--- |
| **Clinical Baseline Model (MLP)** | $\ge$ 60% | **77.5%** | **0.89** | **0.73** |
| **Image Texture Model (EfficientNetB0)** | $\ge$ 70% | **79.0%** | **0.76** | **0.497** |
| **Integrated Multimodal Model (Fusion)** | $\ge$ 75% | **82.0%** | **0.90** | **0.82** |

### Clinical Optimization Strategy
To align the engineering implementation with patient safety priorities, a major emphasis was placed on the **suppression of False Negative errors** (high-risk preterm cases misclassified as healthy pregnancies). 
* Models were trained using dynamically calculated, class-balanced backpropagation weights to counteract the natural dataset imbalance.
* The integrated model utilizes `BinaryFocalCrossentropy` ($\gamma = 2.0, \alpha = 0.85$) to heavily penalize minority class errors and force gradient updates to focus on hard-to-classify patterns.
* The standalone image model loops over confidence intervals during validation to select an optimized operational threshold that limits missed preterm cases (enforcing False Negatives $\le 2$).

---

## Installation & Setup

### Prerequisites
Ensure you have MATLAB installed for the image preprocessing script, and Python 3.8+ for the deep learning models.

### Environment Configuration
To clone the repository and install all necessary Python dependencies directly via your terminal, execute the following commands:

```bash
# Clone the repository
git clone [https://github.com/your-username/Premature-Labor-Risk-through-Cervical-Ultrasound-Imaging.git](https://github.com/your-username/Premature-Labor-Risk-through-Cervical-Ultrasound-Imaging.git)
cd Premature-Labor-Risk-through-Cervical-Ultrasound-Imaging

# Install the required software libraries
pip install tensorflow keras pandas numpy opencv-python scikit-learn matplotlib seaborn joblib openpyxl
```
## How to Train & Evaluate

To replicate the system findings and execute the modules:
First, update the input and output directories in all scripts. Then, follow this sequence:

### Step 1: ROI Extraction & Preprocessing
1. Open MATLAB and run the script located at `preprocessing/roi_processor.m`.
2. Select your raw ultrasound image directory via the graphical popup interface.
3. Use the toolbar zoom to locate the machine’s distance indicator calibration banner, press any key on your keyboard to activate the pointer, and click 2 separate points. Input the true distance in centimeters (cm) when prompted in the MATLAB command window.
4. Locate the cervical tissue using the zoom function, press any key, and click 4 bounding points around the cervical anatomy.
5. The script automatically calculates the rotation angle ($\theta$), corrects the alignment horizontally, crops a normalized region representing 2 cm of tissue depth, and saves the calibrated grayscale ROI image.

### Step 2: Training the Standalone Baselines
* **Tabular Clinical Model:** Execute the clinical Python script to ingest the patient history matrices, handle missing values via global median imputation, compute balanced class weights, and train the Multi-Layer Perceptron (MLP) architecture:
  ```bash
  python clinical_model/clinical_model.py
  ```

* **Image Texture Model:** Execute the computer vision script to load the normalized grayscale crops, scale them to $224 \times 224 \times 3$ formats, initialize the fixed `EfficientNetB0` feature extractor, and compute the optimized validation threshold targeting a maximum of 2 missed preterm cases (False Negatives $\le 2$):
  ```bash
  python image_model/image_processing_model.py
  ```
### Step 3: Training the Integrated Multimodal Model
* Run the early-fusion integration script to structurally align patient ultrasound image tensors with their matching tabular clinical parameters. This constructs the multi-input neural network and trains the shared fusion layers using focal loss optimization:
  ```bash
  python multimodal_fusion/multimodal_model.py
  ```
## Project Outputs & Artifacts

Upon complete execution of the training modules, the system generates the following validated clinical and engineering files:

1. **Model Performance Reports (`.txt`):** Structured summary sheets containing final classification reports (Precision, Sensitivity/Recall, Specificity, and F1-Scores broken down head-to-head for both Term and Preterm outcomes).
2. **Loss and Accuracy Diagnostics (`.png`):** Training history tracking plots charting Train/Validation Accuracy and Train/Validation Loss curves across the active training epochs to document gradient stability.
3. **Confusion Matrices (`.png`):** Detailed 2x2 categorical classification heatmaps outlining the exact True Negative, False Positive, False Negative, and True Positive distributions on the testing partitions.
4. **Diagnostic Metrics Graphs (`.png`):** Updated Receiver Operating Characteristic (ROC) curves charting true positive rates against false positive rates, alongside matching Precision-Recall (PR) curves validating the performance of minority class rankings.
5. **Serialized Weights Files (`.keras` / `.pkl`):** Fully trained, serialized model binaries (`early_labor_model_weighted.keras`, `multimodal_model.keras`) alongside matching dataset standardizers (`scaler_weighted.pkl`) ready for integration or inference deployment.
6. **Diagnostic Explanations (`.png`):** Model interpretability outputs including horizontal feature importance bar charts (mapping clinical drop-in-accuracy risk drivers) and backpropagated Grad-CAM activation heatmaps confirming localized network focus on inner cervical tissue morphology.
