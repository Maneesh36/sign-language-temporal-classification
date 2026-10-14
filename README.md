# Sign Language Recognition Using Temporal Classification

## Project Overview
This project explores sign language recognition using temporal classification on the Auslan sign language dataset. It compares traditional machine learning models and neural network approaches for classifying 95 sign classes.

## Objectives
- Preprocess high-quality and low-quality sign language recordings.
- Compare Logistic Regression, Support Vector Machine (SVM), MLP, and LSTM approaches.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Demonstrate predictions on held-out test recordings.

## Datasets
Two datasets were used:

| Dataset | Recordings | Features | Sequence Length |
|---|---:|---:|---:|
| High-quality | 2,565 | 22 | 136 |
| Low-quality | 6,648 valid recordings | 8 | 100 after resampling |

The data was split into training and test sets using an 80:20 stratified split with random seed 42.

## Models and Results

| Dataset | Model | Accuracy | Macro F1-score |
|---|---|---:|---:|
| High-quality | MLP | 96.49% | 96.55% |
| High-quality | Logistic Regression | 95.71% | 95.67% |
| High-quality | SVM | 74.07% | 74.91% |
| Low-quality | SVM | 64.14% | 63.91% |
| Low-quality | Logistic Regression | 46.92% | 46.38% |

The MLP achieved the highest accuracy on the high-quality dataset. LSTM experiments were also explored but did not achieve satisfactory performance.

## Prediction Demo
The notebook includes an interactive demo that lets users select a held-out test recording and compare the actual sign label with the model's prediction.

**Note:** The demo uses existing labelled recordings; it does not perform live webcam recognition.

## Limitations
- The demo operates on preprocessed test recordings rather than arbitrary video or webcam input.
- The train/test split is recording-level, not signer-independent.
- LSTM performance was limited in the experiments.

## Technologies
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Google Colab
- Jupyter Notebook

## Team
This project was completed collaboratively by two team members.

- Team Member 1: [Sanapala Venkata Maneesh (PES1UG24AM247)]
- Team Member 2: [Shristi Shukla (PES1UG24AM268)]

Update this section to reflect each member's actual contributions.

## Reference
Cate, H., Dalvi, F., & Hussain, Z. (2015). *Sign Language Recognition using Temporal Classification*. Stanford CS229 project report.
