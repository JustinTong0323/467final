import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

def plot_with_labels(X, y, title):
    plt.figure(figsize=(10, 8))
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], label=label, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize features for SER')
    parser.add_argument('--features_path', type=str, required=True, help='Path to the features file')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels file')
    args = parser.parse_args()

    # Load features and labels
    X = np.load(args.features_path)
    y = np.load(args.labels_path)

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plot_with_labels(X_pca, y, 'PCA of Features')

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plot_with_labels(X_tsne, y, 't-SNE of Features')

if __name__ == "__main__":
    main()
