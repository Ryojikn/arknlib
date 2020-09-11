import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

class FeatureExtractor():
    def __init__(self):
        pass

    def vectorizer(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        temp_df['features'] = temp_df[cols[0]]
        for i in range(0, len(cols)-1):
            temp_df['features'] = temp_df['features'] + temp_df[cols[i+1]]
        return temp_df

    def untokenize(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        for col in cols:
            temp_df[col] = temp_df[col].apply(lambda x: ' '.join(word for word in x))
        return temp_df

    class Tfidf(TfidfVectorizer):
        def __init__(self):
            super().__init__()
        

    class W2V():
        def __init__(self):
            pass

        def load_pretrained_model():
            pass