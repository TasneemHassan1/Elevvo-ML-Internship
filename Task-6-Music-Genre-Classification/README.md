# Task 6 - Music Genre Classification

## Overview

This project builds a music genre classification system using the GTZAN dataset.

The system extracts audio features from music files using Librosa and uses machine learning models to classify songs into different music genres.

The extracted features include MFCCs, chroma features, spectral features, zero-crossing rate, RMS energy, and tempo.

Several classification models were trained and evaluated, and the final model was selected based on its performance on the test set.

---

## Dataset

The project uses the **GTZAN Music Genre Dataset**.

The dataset contains music files from 10 different genres:

* Blues
* Classical
* Country
* Disco
* HipHop
* Jazz
* Metal
* Pop
* Reggae
* Rock

The original dataset contains 100 audio files for each genre.

One audio file, `jazz.00054.wav`, could not be processed because its format was not recognized by the audio loading library. The file was documented as a failed file and excluded from the extracted feature dataset.

After feature extraction, **999 audio files** were successfully processed.

---

## Features

Audio features were extracted using **Librosa**.

The extracted features include:

* 13 MFCC coefficients
* Chroma features
* Spectral centroid
* Spectral bandwidth
* Spectral rolloff
* Zero-crossing rate
* RMS energy
* Tempo

For the time-based audio features, the mean and standard deviation were calculated where applicable.

This resulted in **61 numerical audio features** for each successfully processed song.

---

## Data Preparation

The following preprocessing steps were performed:

1. Audio files were loaded using Librosa.
2. Audio features were extracted from each song.
3. Failed audio files were recorded separately.
4. The extracted features were stored in a DataFrame.
5. Missing values were checked.
6. The genre labels were encoded using `LabelEncoder`.
7. The data was divided into training and test sets using an 80/20 split.
8. Stratified splitting was used to preserve the genre distribution.
9. `StandardScaler` was fitted on the training data and then applied to the test data.

The final feature matrix contains **999 samples and 61 features**.

---

## Models

Several machine learning models were trained and compared:

### Logistic Regression

Logistic Regression was used as a baseline classification model.

### Random Forest

A Random Forest classifier with 200 trees was trained using the extracted features.

### SVM

An SVM with an RBF kernel was trained using the scaled features.

### Tuned SVM

GridSearchCV was used to tune the SVM hyperparameters using 5-fold cross-validation on the training data.

The test set was kept separate during the tuning process.

The best parameters were:

```text
C = 100
gamma = scale
kernel = rbf
```

---

## Results

The models were evaluated using accuracy, macro precision, macro recall, and macro F1-score.

| Model               | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| ------------------- | -------: | --------------: | -----------: | -------------: |
| Logistic Regression |    66.5% |          66.92% |        66.5% |         65.95% |
| Random Forest       |    66.5% |          66.68% |        66.5% |         66.04% |
| SVM                 |    68.5% |          68.66% |        68.5% |         68.26% |
| Tuned SVM           |    70.5% |          70.56% |        70.5% |         70.14% |

The tuned SVM achieved the highest test performance among the tested models, with **70.5% accuracy** and a **70.14% macro F1-score**.

The confusion matrix showed that some genres, such as Classical and Jazz, were classified more consistently, while genres such as Rock, Country, Disco, and HipHop had more confusion between them.

---

## Final Model

The tuned SVM was selected as the final model.

The saved model artifact contains:

* Trained SVM model
* StandardScaler
* LabelEncoder
* Feature names

The artifact is saved as:

```text
artifacts/music_genre_model.pkl
```

The saved artifact was also reloaded and tested successfully on the test data.

---

## Project Structure

```text
Task-6-Music-Genre-Classification/
│
├── data/
│   ├── genres_original/
│   │   ├── blues/
│   │   ├── classical/
│   │   ├── country/
│   │   ├── disco/
│   │   ├── hiphop/
│   │   ├── jazz/
│   │   ├── metal/
│   │   ├── pop/
│   │   ├── reggae/
│   │   └── rock/
│   │
│   └── extracted_features.csv
│
├── artifacts/
│   └── music_genre_model.pkl
│
├── music_genre_classification.ipynb
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Librosa
* Scikit-learn
* Matplotlib
* Joblib
* Jupyter Notebook

---

## How to Run

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook music_genre_classification.ipynb
```

### 3. Run the notebook cells

The notebook covers:

* Dataset inspection
* Audio loading
* Feature extraction
* Data preparation
* Model training
* Model evaluation
* SVM hyperparameter tuning
* Model comparison
* Saving and testing the final model

---

## Conclusion

This project demonstrates how audio features can be used with traditional machine learning models to classify music into different genres.

Librosa was used to extract meaningful audio features, while several machine learning models were compared.

The tuned SVM achieved the highest test performance among the tested models with **70.5% accuracy** and **70.14% macro F1-score**.

The final model and preprocessing components were saved together so they can be reused for future predictions.
