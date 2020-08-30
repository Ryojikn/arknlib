import pandas as pd

class FeatureExtractor():
    def __init__(self):
        pass

    def vectorizer(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        temp_df['features'] = temp_df[cols[0]]
        for i in range(0, len(cols)-1):
            temp_df['features'] = temp_df['features'] + temp_df[cols[i+1]]
        return temp_df

    def w2v_fit(self, df: pd.DataFrame, feature_col: str):
        temp_df = df.copy()
        return model, model_vocab
    
    def w2v_transform(self, df: pd.DataFrame, col: str, w2v_model):
        temp_df = df.copy()

        return temp_df
    
