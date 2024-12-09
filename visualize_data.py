import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distribution(data_path, save_path='emotion_distribution.png'):
    # Load dataset
    data_df = pd.read_csv(data_path)
    
    # Plot the distribution of emotions
    plt.figure(figsize=(10, 6))
    sns.countplot(x='emotion', data=data_df, order=data_df['emotion'].value_counts().index)
    plt.title('Distribution of Emotions in the Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Save the plot
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    visualize_distribution('all_df.csv')
