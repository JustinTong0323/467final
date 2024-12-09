import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    

    def extract_features_for_df(self, df, augment=False, save_path=None):
        features = []
        labels = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Extracting features'):
            feature = self.extract_feature(row['file_path'], augment=augment)
            features.extend(feature)
            labels.append(row['label'])
        features = np.array(features)
        labels = np.array(labels).reshape(-1, 1)
        
        if save_path:
            np.save(os.path.join(save_path, 'features.npy'), features)
            np.save(os.path.join(save_path, 'labels.npy'), labels)
        
        return features, labels
    
    def visualize_features(self, features, labels, save_path=None, method='pca'):
        import matplotlib.pyplot as plt

        # Reduce dimensions to 2D
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2)
        else:
            raise ValueError("Unsupported method. Choose from 'pca', 'tsne', or 'umap'.")

        reduced_features = reducer.fit_transform(features)

        # Plot the features
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            indices = np.where(labels == label)
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=label)
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'2D Visualization of Features using {method.upper()}')
        plt.legend()

        if save_path:
            plt.savefig(os.path.join(save_path, f'features_{method}.png'))
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Feature extraction for SER')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the extracted features')
    parser.add_argument('--enable_augmentation', action='store_true', help='Enable data augmentation')
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data_path)

    # Extract features
    feature_extractor = FeatureExtractor()
    # features, labels = feature_extractor.extract_features_for_df(df, augment=args.enable_augmentation, save_path=args.save_path)
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    emotion_list = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'calm']
    labels = np.array([emotion_list[int(label)] for label in labels])
    feature_extractor.visualize_features(features, labels, args.save_path, method='pca')
    feature_extractor.visualize_features(features, labels, args.save_path, method='tsne')
    

if __name__ == "__main__":
    main()
