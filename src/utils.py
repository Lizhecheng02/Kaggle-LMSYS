import pandas as pd
import json
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def unicode_preprocess(input_str):
    return json.loads(input_str)


def classification_data_preprocessing(df_path):
    df = pd.read_csv(df_path)

    df.loc[:, "prompt"] = df["prompt"].apply(unicode_preprocess)
    df.loc[:, "response_a"] = df["response_a"].apply(unicode_preprocess)
    df.loc[:, "response_b"] = df["response_b"].apply(unicode_preprocess)

    df.drop(columns=["model_a", "model_b"], axis=1, inplace=True)
    df.dropna(inplace=True)
    df["label"] = df.apply(lambda row: 0 if row["winner_model_a"] == 1 else (1 if row["winner_model_b"] == 1 else 2), axis=1)
    df.drop(columns=["winner_model_a", "winner_model_b", "winner_tie"], axis=1, inplace=True)

    full_chat_a = []
    full_chat_b = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]
        response_a = row["response_a"]
        response_b = row["response_b"]
        chat_a = ""
        chat_b = ""
        for i in range(len(prompt)):
            prompt[i] = prompt[i] or ""
            response_a[i] = response_a[i] or ""
            response_b[i] = response_b[i] or ""
            if i == 0 and i != len(prompt) - 1:
                chat_a += prompt[i] + " " + response_a[i] + "\n"
                chat_b += prompt[i] + " " + response_b[i] + "\n"
            elif i == 0 and i == len(prompt) - 1:
                chat_a += prompt[i] + " " + response_a[i]
                chat_b += prompt[i] + " " + response_b[i]
            elif i == len(prompt) - 1:
                chat_a += prompt[i] + " " + response_a[i]
                chat_b += prompt[i] + " " + response_b[i]
            else:
                chat_a += prompt[i] + " " + response_a[i] + "\n"
                chat_b += prompt[i] + " " + response_b[i] + "\n"
        full_chat_a.append(chat_a)
        full_chat_b.append(chat_b)

    df["full_chat_a"] = full_chat_a
    df["full_chat_b"] = full_chat_b

    train, val = train_test_split(df, test_size=0.1, random_state=3407)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    print("All Column Names:", train.columns)

    return train, val


if __name__ == "__main__":
    train, val = classification_data_preprocessing("../data/train.csv")
