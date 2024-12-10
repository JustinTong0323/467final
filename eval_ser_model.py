import pandas as pd
from ser_wav2vec import SpeechEmotionRecognition

df_train_val = pd.read_csv('train_val_df.csv')
df_test = pd.read_csv('test_df.csv')

args = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 30,
    'gradient_accumulation_steps': 4,
    'checkpoint_dir': 'checkpoints',
    'checkpoint': 'best_model.pt'
}

ser = SpeechEmotionRecognition(
    df=df_train_val,
    model_name='jonatasgrosman/wav2vec2-large-xlsr-53-english',
    batch_size=args['batch_size'],
    lr=args['lr'],
    num_epochs=args['epochs'],
    checkpoint_dir=args['checkpoint_dir'],
    gradient_accumulation_steps=args['gradient_accumulation_steps'],
    checkpoint_path=args['checkpoint']
)

ser.evaluate_test_set(df_test)