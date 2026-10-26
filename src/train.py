import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

df = pd.read_parquet('train_processed.parquet')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

df['unique_words'] = df['text'].apply(lambda x: len(set(x.lower().split())))

X_text = tfidf.transform(df['text'])
X_num = df[['msg_count', 'avg_len', 'emoji_count', 'unique_words', 'echo_score', 'eng_count']]
y = df['is_bot']

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
    X_text, X_num_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train = hstack([X_text_train, X_num_train]).tocsr()
X_val = hstack([X_text_val, X_num_val]).tocsr()

mlflow.set_experiment("YouAreBot_Final")

with mlflow.start_run():
    C = 1.0
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_param("features", "tfidf_1000_bigrams + 6_num_features")
    
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, y_pred_proba)
    
    mlflow.log_metric("val_logloss", loss)
    print(f"Validation LogLoss: {loss:.4f}")
    
    with open('baseline_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
