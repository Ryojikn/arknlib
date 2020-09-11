import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer, SnowballStemmer
from string import punctuation
from unicodedata import normalize, combining
import unicodedata as ud

class Prep:
    def __init__(self, language='english'):
        self.language = language
        self.stopwords_list = stopwords.words(self.language)

    def clean_whitespaces(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        for col in cols:
            if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                temp_df[col] = temp_df[col].apply(lambda x: " ".join(x.split()))
        return temp_df
    
    def remove_punctuation(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        for col in cols:
            if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                temp_df[col] = temp_df[col].apply(lambda x: x.translate(str.maketrans('', '', punctuation)))
        return temp_df

    def remove_accent(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        for col in cols:
            if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                temp_df[col] = temp_df[col].apply(lambda x: u"".join([c for c in normalize('NFKD', x) if not combining(c)]))
        return temp_df
    
    def to_lower(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        for col in cols:
            if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                temp_df[col] = temp_df[col].apply(lambda x: x.lower())
        return temp_df

    def replace_regex(self, df: pd.DataFrame, cols: list,
                            regex_list: list=None, replace_str:str=None):
        if regex_list is None:
            regex_list = []
        if replace_str is None:
            replace_str = ''
        temp_df = df.copy()
        for col in cols:
            for regex in regex_list:
                if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                    temp_df[col] = temp_df[col].apply(lambda x: re.sub(regex, replace_str, x))
        return temp_df
    
    def remove_stopwords(self, df: pd.DataFrame, cols: list, stopwords_list:list = None):
        if stopwords_list is None:
            stopwords_list = self.stopwords_list
        if self.language == 'portuguese':
            removal_list = ['não', 'num']
            self.stopwords_list.remove(x for x in removal_list)
        temp_df = df.copy()
        for col in cols:
                if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                    temp_df[col] = temp_df[col].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords_list]))
        return temp_df

    def stemming(self, df: pd.DataFrame, cols: list, stemmer='snowball', ignore_stopwords=False):
        '''
        Parameters:
        df: pandas DataFrame
        cols: list with the name of columns that you want to run the pipeline
        stemmer: choose between snowball or rslp
        ignore_stopwords: only used for snowball
        '''
        temp_df = df.copy()
        if stemmer == 'snowball':
            stemmer = SnowballStemmer(self.language, ignore_stopwords)
        elif stemmer == 'rslp':
            stemmer = RSLPStemmer()
        for col in cols:
                if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                    temp_df[col] = temp_df[col].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
        return temp_df
        

    def lemmatize(self, df: pd.DataFrame, cols: list):
        pass

    def anonymize(self, df: pd.DataFrame, cols: list):
        pass

    def tokenize(self, df: pd.DataFrame, cols: list):
        temp_df = df.copy()
        for col in cols:
            if (temp_df[col].dtype == object) and (type(temp_df[col].iloc[0]) == str):
                temp_df[col] = temp_df[col].apply(lambda x: word_tokenize(x, self.language))
        return temp_df

    def pipeline(self, df: pd.DataFrame, cols: list, pipeline=[3],
                regex_list: list = None, replace_str: str = '', stopwords_list: list = None):
        '''
        Parameters:
        df: pandas DataFrame
        cols: list with the name of columns that you want to run the pipeline
        pipeline: list containing order of the functions, be aware that tokenize if selected, 
        must be in the final position of the list.
        0 = clean_whitespaces
        1 = remove_punctuation
        2 = remove_accent
        3 = to_lower
        4 = replace_regex
        5 = remove_stopwords
        6 = stemming
        7 = lemmatize
        8 = anonymize
        9 = tokenize

        Returns: 
        temp_df = pandas DataFrame
        '''
        functions = {
            0: self.clean_whitespaces,
            1: self.remove_punctuation,
            2: self.remove_accent,
            3: self.to_lower,
            4: self.replace_regex,
            5: self.remove_stopwords,
            6: self.stemming,
            7: self.lemmatize,
            8: self.anonymize,
            9: self.tokenize
        }
        temp_df = df.copy()
        for item in pipeline:
            if item == 4:
                temp_df = functions[item](temp_df, cols, regex_list, replace_str)
            elif item == 5:
                temp_df = functions[item](temp_df, cols, stopwords_list)
            else:
                temp_df = functions[item](temp_df, cols)
        
        return temp_df
