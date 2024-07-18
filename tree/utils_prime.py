from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from config import CFG
import pandas as pd
import pickle
import json
import os


def tokenizer(x):
    return x


def preprocessor(x):
    return x


def process(input_str):
    return json.loads(input_str)


def load_json(data):
    data.loc[:, "prompt"] = data["prompt"].apply(process)
    data.loc[:, "response_a"] = data["response_a"].apply(process)
    data.loc[:, "response_b"] = data["response_b"].apply(process)
    return data


def use_tf_idf_vectorizer(train, column_name, idx):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        token_pattern=None,
        strip_accents="unicode",
        analyzer="word",
        sublinear_tf=True,
        ngram_range=CFG.tfidf_vectorizer_ngram_range,
        min_df=CFG.tfidf_vectorizer_min_df,
        max_df=CFG.tfidf_vectorizer_max_df,
        max_features=CFG.tfidf_vectorizer_max_features
    )

    X_train = vectorizer.fit_transform([i for i in train[column_name]])
    dense_matrix = X_train.toarray()
    df = pd.DataFrame(dense_matrix)
    tfidf_columns = [f"{column_name}_tfidf_{i}" for i in range(len(df.columns))]
    df.columns = tfidf_columns
    df["id"] = train["id"]

    if not os.path.exists("vectorizers"):
        os.makedirs("vectorizers")

    with open(f"./vectorizers/tf_idf_vectorizer_{column_name}_{idx}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer, df


def use_count_vectorizer(train, column_name, idx):
    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        token_pattern=None,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=CFG.count_vectorizer_ngram_range,
        min_df=CFG.count_vectorizer_min_df,
        max_df=CFG.count_vectorizer_max_df,
    )

    X_train = vectorizer.fit_transform([i for i in train[column_name]])
    dense_matrix = X_train.toarray()
    df = pd.DataFrame(dense_matrix)
    count_columns = [f"{column_name}_count_{i}" for i in range(len(df.columns))]
    df.columns = count_columns
    df["id"] = train["id"]

    if not os.path.exists("vectorizers"):
        os.makedirs("vectorizers")

    with open(f"./vectorizers/count_vectorizer_{column_name}_{idx}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer, df


def use_tf_idf_vectorizer_for_test(vectorizer, test, column_name, idx):
    X_test = vectorizer.transform([i for i in test[column_name]])
    dense_matrix = X_test.toarray()
    df = pd.DataFrame(dense_matrix)
    tfidf_columns = [f"{column_name}_tfidf_{i}_{idx}" for i in range(len(df.columns))]
    df.columns = tfidf_columns
    df["id"] = test["id"]
    return df


def use_count_vectorizer_for_test(vectorizer, test, column_name, idx):
    X_test = vectorizer.transform([i for i in test[column_name]])
    dense_matrix = X_test.toarray()
    df = pd.DataFrame(dense_matrix)
    count_columns = [f"{column_name}_count_{i}_{idx}" for i in range(len(df.columns))]
    df.columns = count_columns
    df["id"] = test["id"]
    return df
