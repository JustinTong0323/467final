# CSCI 467 final project

This repository contains all the code and data used for the Speech Emotion Recognition (SER) project. The project aims to classify emotions from speech audio using machine learning models.

## Before You Begin

Install the required packages using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

## Datasets

The following publicly available datasets are used:

1. **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
   - [Download Link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

2. **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
   - [Download Link](https://www.kaggle.com/datasets/ejlok1/cremad)

3. **TESS (Toronto Emotional Speech Set)**
   - [Download Link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

4. **SAVEE (Surrey Audio-Visual Expressed Emotion)**
   - [Download Link](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)

Please download these datasets and place them in a `data` directory with the following structure:

```
data/
├── RAVDESS/
├── CREMA-D/
├── TESS/
└── SAVEE/
```

## Use Jupyter Notebook Directly

You can directly open the `report.ipynb` file using Jupyter Notebook and run the cells to generate the results.

## Commands to Generate Results Step by Step

1. **Combine and Prepare Data**

   ```bash
   python combine_data.py
   ```

   - Loads and combines datasets into `all_df.csv`.
   - Splits data into training, validation, and test sets.
   - Saves the splits as `train_val_df.csv` and `test_df.csv`.

 

2. **Train Baseline Models**

   ```bash
   python feature_extractor_and_baseline_model.py --enable_argument
   ```

   - Extracts features from audio files with data augmentation (`--enable_argument` flag) or without data augmentation.
   - Trains Logistic Regression and Random Forest models.
   - Evaluates models on the test set.

3. **Train the SER Model**

    Adjust the hyperparameters in `train_ser_model.py` as needed. Then, run:

    ```bash
    python train_ser_model.py
    ```

    - Trains the speech emotion recognition model using specified model architecture and hyperparameters.
    - Uses `train_val_df.csv` for training and validation.
    - Saves the model checkpoint per epoch in the `checkpoints` directory(could be changed in the script).

4. **Evaluate the SER Model**

   ```bash
   python eval_ser_model.py
   ```

   - Evaluates the trained SER model on the test set (`test_df.csv`).
   - Print classification report with accuracy and F1 score. Also, save the confusion matrix.