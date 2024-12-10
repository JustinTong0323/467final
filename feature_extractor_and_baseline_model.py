import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import argparse


class FeatureExtractor:
    def __init__(self, target_sr=16000, n_mfcc=40):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        return y, sr

    def normalize_audio(self, y):
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y_normalized = y / rms
        else:
            y_normalized = y
        return y_normalized

    def augment_audio(self, y, sr):
        augmented_data = []
        
        # Original
        augmented_data.append(y)
        
        # Add noise
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        augmented_data.append(y_noise)
        
        # Time stretching
        y_stretch = librosa.effects.time_stretch(y, rate=0.9)
        augmented_data.append(y_stretch)
        y_stretch = librosa.effects.time_stretch(y, rate=1.1)
        augmented_data.append(y_stretch)
        
        # Pitch shifting
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented_data.append(y_shift)
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        augmented_data.append(y_shift)
        
        # Reverberation (simple simulation)
        y_reverb = librosa.effects.preemphasis(y)
        augmented_data.append(y_reverb)
        
        return augmented_data

    def extract_feature(self, file_path, augment=False):
        y, sr = self.load_audio(file_path)
        y = self.normalize_audio(y)
        
        if augment:
            augmented_audios = self.augment_audio(y, sr)
        else:
            augmented_audios = [y]
        
        features = []
        for augmented_y in augmented_audios:
            # Ensure consistent length by trimming or padding
            max_length = self.target_sr * 3  # 3 seconds
            if len(augmented_y) > max_length:
                augmented_y = augmented_y[:max_length]
            else:
                augmented_y = np.pad(augmented_y, (0, max_length - len(augmented_y)), mode='constant')
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=augmented_y, sr=sr, n_mfcc=self.n_mfcc)
            mfccs = np.mean(mfccs.T, axis=0)
            
            # Extract additional features if needed
            chroma = librosa.feature.chroma_stft(y=augmented_y, sr=sr)
            chroma = np.mean(chroma.T, axis=0)
            spectral_contrast = librosa.feature.spectral_contrast(y=augmented_y, sr=sr)
            spectral_contrast = np.mean(spectral_contrast.T, axis=0)
            
            # Concatenate all features
            feature_vector = np.concatenate([mfccs, chroma, spectral_contrast])
            features.append(feature_vector)
            
        # Flatten the features
        feature = np.array(features).flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        feature = feature.reshape(1, -1)
        
        return feature

    def extract_features_for_df(self, df, augment=False):
        features = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Extracting features'):
            feature = self.extract_feature(row['file_path'], augment=augment)
            features.extend(feature)
        return np.array(features)

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, f1

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, f1

def main():
    parser = argparse.ArgumentParser(description='Baseline model for SER')
    parser.add_argument('--enable_argument', action='store_true', help='Enable this argument')
    args = parser.parse_args()
    
    # Load datasets
    train_val_df = pd.read_csv('train_val_df.csv')
    test_df = pd.read_csv('test_df.csv')

    # Extract features
    feature_extractor = FeatureExtractor()
    X_train_val = feature_extractor.extract_features_for_df(train_val_df, augment=args.enable_argument)
    X_test = feature_extractor.extract_features_for_df(test_df, augment=args.enable_argument)

    # Get labels
    y_train_val = train_val_df['label'].values
    y_test = test_df['label'].values

    # Train and evaluate Logistic Regression model
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train_val, y_train_val)
    test_accuracy_lr, test_f1_lr = lr_model.evaluate(X_test, y_test)
    print(f'Logistic Regression - Test Accuracy: {test_accuracy_lr:.4f}, F1 Score: {test_f1_lr:.4f}')

    # Train and evaluate Random Forest model
    rf_model = RandomForestModel()
    rf_model.train(X_train_val, y_train_val)
    test_accuracy_rf, test_f1_rf = rf_model.evaluate(X_test, y_test)
    print(f'Random Forest - Test Accuracy: {test_accuracy_rf:.4f}, F1 Score: {test_f1_rf:.4f}')

if __name__ == "__main__":
    main()
