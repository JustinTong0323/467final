import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2ForSequenceClassification,
)
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import argparse

class SpeechEmotionRecognition:
    def __init__(self, df, model_name='facebook/wav2vec2-base', batch_size=2, lr=1e-5, num_epochs=5, checkpoint_dir='checkpoints', gradient_accumulation_steps=4, checkpoint_path=None):  # Reduced batch size to mitigate memory issues
        self.df = df
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_path

        # Initialize processor
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate

        # Split data
        self.train_df, self.val_df = train_test_split(
            self.df, test_size=0.2, stratify=self.df['label'], random_state=42
        )

        # Create datasets and dataloaders
        self.train_dataset = self.SpeechEmotionDataset(self.train_df, self.target_sampling_rate)
        self.val_dataset = self.SpeechEmotionDataset(self.val_df, self.target_sampling_rate)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.speech_collate_fn, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.speech_collate_fn, num_workers=4, pin_memory=True)

        # Initialize model
        self.model = self.create_model()

        # Load from checkpoint if provided
        if self.checkpoint_path is not None:
            try:
                self.model.load_state_dict(torch.load(self.checkpoint_path))
                print(f"Loaded model from checkpoint: {self.checkpoint_path}")
            except (FileNotFoundError, RuntimeError) as e:
                print(f"Error loading checkpoint: {e}")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    class SpeechEmotionDataset(Dataset):
        def __init__(self, dataframe, target_sampling_rate=16000):
            self.dataframe = dataframe.reset_index(drop=True)
            self.target_sampling_rate = target_sampling_rate

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            audio_file = self.dataframe.loc[idx, 'file_path']
            label = self.dataframe.loc[idx, 'label']

            # Load audio
            speech_array, sampling_rate = librosa.load(audio_file, sr=None)

            # Resample if needed
            if sampling_rate != self.target_sampling_rate:
                speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=self.target_sampling_rate)

            # Convert to mono if stereo
            if len(speech_array.shape) > 1:
                speech_array = librosa.to_mono(speech_array)

            return {"speech": torch.tensor(speech_array, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}

    def speech_collate_fn(self, batch):
        max_length = max([item['speech'].shape[0] for item in batch])
        speech_list = [torch.nn.functional.pad(item['speech'], (0, max_length - item['speech'].shape[0])) for item in batch]
        labels = [item['labels'] for item in batch]

        inputs = self.processor(
            speech_list,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        labels = torch.tensor(labels, dtype=torch.long)
        inputs['labels'] = labels

        return inputs

    def create_model(self):
        num_labels = len(self.df['label'].unique())

        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            finetuning_task="classification",
        )
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory

        return model

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 10)

            # Training
            self.model.train()
            train_loss = 0.0

            for i, batch in enumerate(tqdm(self.train_loader, desc="Training")):
                input_values = batch['input_values'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                # Ensure the input is 2D (batch_size, sequence_length)
                input_values = input_values.squeeze() if len(input_values.shape) == 3 else input_values

                outputs = self.model(
                    input_values=input_values,
                    labels=labels
                )

                loss = outputs.loss

                # Ensure the loss is scalar
                loss = loss.mean()
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * self.gradient_accumulation_steps

            avg_train_loss = train_loss / len(self.train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            # Validation
            self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_values = batch['input_values'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                # Ensure the input is 2D (batch_size, sequence_length)
                input_values = input_values.squeeze() if len(input_values.shape) == 3 else input_values

                outputs = self.model(
                    input_values=input_values,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # Ensure the loss is scalar
                loss = loss.mean()

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = correct / total

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    def predict(self, audio_file):
        self.model.eval()

        speech_array, sampling_rate = librosa.load(audio_file, sr=None)

        if sampling_rate != self.processor.feature_extractor.sampling_rate:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=self.processor.feature_extractor.sampling_rate)

        if len(speech_array.shape) > 1:
            speech_array = librosa.to_mono(speech_array)

        inputs = self.processor(
            speech_array,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()

        return predicted_class_id

    def evaluate_test_set(self, test_df):
        test_dataset = self.SpeechEmotionDataset(test_df, self.target_sampling_rate)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.speech_collate_fn, num_workers=4, pin_memory=True)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_values = batch['input_values'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                # Ensure the input is 2D (batch_size, sequence_length)
                input_values = input_values.squeeze() if len(input_values.shape) == 3 else input_values

                outputs = self.model(input_values=input_values)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True, help='Path to the CSV file containing file paths and labels')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint to continue training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--test_csv', type=str, default=None, help='Path to the CSV file containing test data for evaluation')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_csv)

    # Initialize and train model
    ser = SpeechEmotionRecognition(
        df=df,
        model_name='facebook/wav2vec2-base',
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_path=args.checkpoint
    )
    ser.train()

    # Evaluate on test set if provided
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        ser.evaluate_test_set(test_df)

if __name__ == "__main__":
    main()
