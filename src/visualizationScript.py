import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv')
    
    # Category distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category', data=df)
    plt.title('Distribution of Message Categories')
    plt.savefig('data/visualizations/category_distribution.png')
    plt.close()
    
    # Message volume over time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_volume = df.groupby('date').size().reset_index(name='count')
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_volume['date'], daily_volume['count'])
    plt.title('Daily Message Volume')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/visualizations/daily_volume.png')
    plt.close()
    
    from wordcloud import WordCloud
    
    all_text = ' '.join(df['processed_content'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Message Content')
    plt.tight_layout()
    plt.savefig('data/visualizations/wordcloud.png')
    plt.close()

if __name__ == "__main__":
    create_visualizations()
    print("Visualizations created and saved.")