import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

def get_echo_score(user_msgs, other_msgs):
    if len(user_msgs) < 2 or len(other_msgs) == 0:
        return 0.0
    last_user_text = user_msgs[-1]['text'].lower()
    prev_other_text = other_msgs[-1]['text'].lower()
    user_words = set(re.findall(r'\b\w+\b', last_user_text))
    other_words = set(re.findall(r'\b\w+\b', prev_other_text))
    if len(user_words) == 0:
        return 0.0
    return len(user_words.intersection(other_words)) / len(user_words)

def load_and_process(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.read_csv(csv_path)
    
    texts, msg_counts, avg_lens, emoji_counts, echo_scores, eng_counts = [], [], [], [], [], []
    
    for idx, row in df.iterrows():
        msgs = sorted(data[row['dialog_id']], key=lambda x: x['message'])
        user_msgs = [m for m in msgs if m['participant_index'] == row['participant_index']]
        other_msgs = [m for m in msgs if m['participant_index'] != row['participant_index']]
        
        full_text = ' '.join([m['text'] for m in user_msgs])
        
        texts.append(full_text)
        msg_counts.append(len(user_msgs))
        avg_lens.append(sum(len(t) for t in user_msgs) / (len(user_msgs) if user_msgs else 1))
        emoji_counts.append(len([c for c in full_text if not c.isalnum() and not c.isspace()]))
        echo_scores.append(get_echo_score(user_msgs, other_msgs))
        eng_counts.append(len(re.findall(r'[a-zA-Z]', full_text)))
        
    df['text'] = texts
    df['msg_count'] = msg_counts
    df['avg_len'] = avg_lens
    df['emoji_count'] = emoji_counts
    df['echo_score'] = echo_scores
    df['eng_count'] = eng_counts
    
    return df

train_df = load_and_process('datasets/train.json', 'datasets/ytrain.csv')

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, stop_words=None)
tfidf.fit_transform(train_df['text'])

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

train_df.to_parquet('train_processed.parquet', index=False)

