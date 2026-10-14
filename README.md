# Sign Language Recognition Using Temporal Classification

## 1. Project Overview

This project implements sign language recognition using temporal classification on the Auslan sign language dataset. It explores traditional machine learning algorithms and neural network architectures to classify recorded sign sequences into 95 sign classes.

The project investigates the performance of different models on high-quality and low-quality datasets and includes a prediction demo that compares predicted labels with actual labels.

## 2. Objectives

- Preprocess and prepare high-quality and low-quality sign language datasets.
- Implement and evaluate traditional machine learning classifiers.
- Explore neural network models for sign language classification.
- Compare model performance using accuracy, precision, recall, and F1-score.
- Demonstrate predictions on held-out test recordings.

## 3. Models Implemented

The following models were explored:

- Logistic Regression
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)

The traditional models were evaluated on both high-quality and low-quality datasets. The MLP and LSTM experiments focused on the high-quality dataset.

## 4. Dataset Information

The project uses two versions of the Auslan sign language dataset.

### High-Quality Dataset

- Recordings: 2,565
- Features per time step: 22
- Time steps per recording: 136
- Number of classes: 95

### Low-Quality Dataset

- Valid recordings: 6,648
- Selected features per time step: 8
- Resampled sequence length: 100 time steps
- Number of classes: 95

The low-quality dataset contains variable-length sequences. These were resampled to a fixed length for model input. One empty recording was skipped during preprocessing.

### Data Splitting

Both datasets were split into training and testing sets using stratified sampling:

- Training set: 80%
- Testing set: 20%
- Random seed: 42

All 95 classes are represented in both training and testing splits.

**Note:** The split was performed at the recording level, not the signer level. Therefore, the reported results do not establish signer-independent generalization.

## 5. Technologies and Libraries

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- IPyWidgets
- Joblib
- SciPy
- Google Colab
- Google Drive

## 6. Project Structure

The main files in this repository are:

```text
sign-language-temporal-classification/
├── Sign_Language_Temporal_Classification_Final.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook contains the project workflow, including preprocessing, model experiments, evaluation, and the prediction demo.

The datasets, preprocessed arrays, trained models, and experiment outputs are stored separately in Google Drive and are not included in this repository.

## 7. Setup and Execution Instructions

The project was developed and tested using Google Colab.

### Step 1: Open the Notebook

Open the following notebook from this repository:

`Sign_Language_Temporal_Classification_Final.ipynb`

Download it or open it in Google Colab.

### Step 2: Prepare Google Drive

The notebook uses project files stored in Google Drive.

Upload or make the required project files available under:

```text
MyDrive/Sign_Language_Temporal_Classification/
```

Depending on the notebook workflow, these files include:

- `X_high_preprocessed.npy`
- `y_high_preprocessed.npy`
- `X_low_preprocessed.npy`
- `y_low_preprocessed.npy`
- `Xh_train.npy`
- `Xh_test.npy`
- `yh_train.npy`
- `yh_test.npy`
- `Xl_train.npy`
- `Xl_test.npy`
- `yl_train.npy`
- `yl_test.npy`
- `experiment_config.json`
- Required trained models and preprocessing objects from the `results/` directory

The exact files required depend on which experiments or prediction workflow you want to execute.

**Important:** Cloning this GitHub repository alone does not download the datasets or trained models stored in Google Drive.

### Step 3: Mount Google Drive

Run the notebook cell that mounts Google Drive. If a mount cell is needed, use:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorize access when prompted.

### Step 4: Install Dependencies

Google Colab already includes many common Python libraries. If a required library is missing, run:

```python
!pip install numpy pandas scikit-learn tensorflow matplotlib seaborn ipywidgets joblib scipy
```

The `requirements.txt` file lists the project's main Python dependencies.

### Step 5: Load the Datasets and Models

Run the notebook cells responsible for loading the datasets, labels, preprocessing objects, and saved models.

Check that the configured Google Drive paths are correct and that all required files are available.

For the saved MLP prediction workflow, the model expects 2,992 input features per recording, corresponding to the flattened high-quality input of 136 time steps and 22 features.

The prediction workflow must use the same preprocessing applied during training.

### Step 6: Execute the Notebook

Run the notebook cells in their intended order.

The notebook covers:

1. Dataset loading and inspection
2. Data preprocessing and splitting
3. Traditional machine learning experiments
4. Neural network experiments
5. Model evaluation and comparison
6. Prediction demo

If you only want to run the saved-model prediction demo, load the required test data, labels, trained MLP, label encoder, and fitted scaler before executing the demo cell.

## 8. Experimental Results

The following results were obtained on the held-out test sets.

| Dataset | Model | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|---|---:|---:|---:|---:|
| High-quality | MLP | 96.49% | 97.19% | 96.63% | 96.55% |
| High-quality | Logistic Regression | 95.71% | 96.14% | 95.79% | 95.67% |
| High-quality | SVM | 74.07% | 81.04% | 74.60% | 74.91% |
| Low-quality | SVM | 64.14% | 66.14% | 64.14% | 63.91% |
| Low-quality | Logistic Regression | 46.92% | 47.56% | 46.92% | 46.38% |

The MLP achieved the highest test accuracy among the evaluated models, at **96.49%** on the high-quality dataset.

The LSTM experiments were also explored, but they did not achieve competitive classification performance in the experiments conducted.

## 9. Prediction Demo

The notebook includes a dataset-based prediction demo.

It allows a held-out test recording to be selected and displays:

- The actual sign label
- The model's predicted sign label
- Whether the prediction is correct

The demo uses the trained MLP and appropriately preprocessed high-quality test recordings.

**The demo is not a live webcam-based sign language recognition system.** It performs predictions on recordings already present in the dataset.

## 10. Limitations

- The project classifies recorded sign sequences rather than continuous live webcam input.
- The train/test split is recording-level rather than signer-independent.
- The high-quality and low-quality datasets have different feature dimensions and sequence representations.
- LSTM experiments did not achieve competitive performance in the current implementation.
- Reproducing the experiments requires the datasets and trained model files stored separately in Google Drive.

## 11. Reference

Cate, H., Dalvi, F., and Hussain, Z. (2015). *Sign Language Recognition using Temporal Classification.*

Paper: https://cs229.stanford.edu/proj2015/118_report.pdf

---

---

## Team Members

- **Name:** Sanapala Venkata Maneesh | **SRN:** PES1UG24AM247
- **Name:** Shristi Shukla  | **SRN:** PES1UG24AM247


