import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import argparse

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
    parser.add_argument('--train_features_path', type=str, required=True, help='Path to the training features file')
    parser.add_argument('--test_features_path', type=str, required=True, help='Path to the testing features file')
    parser.add_argument('--train_labels_path', type=str, required=True, help='Path to the training labels file')
    parser.add_argument('--test_labels_path', type=str, required=True, help='Path to the testing labels file')
    args = parser.parse_args()

    # Load features and labels
    X_train = np.load(args.train_features_path)
    X_test = np.load(args.test_features_path)
    y_train = np.load(args.train_labels_path)
    y_test = np.load(args.test_labels_path)

    # Train and evaluate Logistic Regression model
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    test_accuracy_lr, test_f1_lr = lr_model.evaluate(X_test, y_test)
    print(f'Logistic Regression - Test Accuracy: {test_accuracy_lr:.4f}, F1 Score: {test_f1_lr:.4f}')

    # Train and evaluate Random Forest model
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    test_accuracy_rf, test_f1_rf = rf_model.evaluate(X_test, y_test)
    print(f'Random Forest - Test Accuracy: {test_accuracy_rf:.4f}, F1 Score: {test_f1_rf:.4f}')

if __name__ == "__main__":
    main()
