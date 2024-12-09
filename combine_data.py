import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data_ravdess(data_dir):
    emotion_labels = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                emotion = emotion_labels.get(parts[2])
                file_list.append({'file_path': os.path.join(root, file), 'emotion': emotion})
    return pd.DataFrame(file_list)

def load_data_cremad(data_dir):
    emotion_labels = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    file_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            parts = file.split('_')
            emotion = emotion_labels.get(parts[2])
            if emotion:
                file_list.append({'file_path': os.path.join(data_dir, file), 'emotion': emotion})
    return pd.DataFrame(file_list)

def load_data_tess(data_dir):
    emotion_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fearful',
        'happy': 'happy',
        'ps': 'surprised',  # Pleasant surprise
        'sad': 'sad',
        'neutral': 'neutral'
    }
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                emotion = file.split('_')[2].split('.')[0]
                emotion_label = emotion_map.get(emotion)
                if emotion_label:
                    file_list.append({'file_path': os.path.join(root, file), 'emotion': emotion_label})
    return pd.DataFrame(file_list)

def load_data_savee(data_dir):
    emotion_map = {
        'a': 'angry',
        'd': 'disgust',
        'f': 'fearful',
        'h': 'happy',
        'n': 'neutral',
        'sa': 'sad',
        'su': 'surprised'
    }
    file_list = []
    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            emotion_code = file.split('_')[1][:2]
            emotion_label = emotion_map.get(emotion_code)
            if emotion_label:
                file_list.append({'file_path': os.path.join(data_dir, file), 'emotion': emotion_label})
    return pd.DataFrame(file_list)

# Load datasets
ravdess_df = load_data_ravdess('data/RAVDESS')
cremad_df = load_data_cremad('data/CREMA-D')
tess_df = load_data_tess('data/TESS')
savee_df = load_data_savee('data/SAVEE')

# Combine datasets
data_df = pd.concat([ravdess_df, cremad_df, tess_df, savee_df], ignore_index=True)

# Standardize emotion labels
emotion_list = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'calm']

# Handle any missing emotions in datasets
data_df = data_df[data_df['emotion'].isin(emotion_list)]

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(emotion_list)
data_df['label'] = label_encoder.transform(data_df['emotion'])

# Save the combined dataset
data_df.to_csv('all_df.csv', index=False)

# Split data into train, validation, and test sets
train_df, temp_df = train_test_split(data_df, test_size=0.3, stratify=data_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Combine train and validation sets
train_val_df = pd.concat([train_df, val_df], ignore_index=True)

# Save the datasets
train_val_df.to_csv('train_val_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

print("Data preparation complete. Files saved as 'all_df.csv', 'train_val_df.csv', and 'test_df.csv'.")
