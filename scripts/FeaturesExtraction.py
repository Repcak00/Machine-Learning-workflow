#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

import nltk
import numpy as np
import pandas as pd
import typer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from DataPreprocessing import load_params
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')

def remove_features(df, col_name_to_ramain):
    ret_df = df[[f"{col_name_to_ramain}"]]
    return ret_df


def add_nan_column(df, col_name):
    df_ret = df.copy(deep=True)
    df_ret[col_name] = np.nan
    return df_ret

def remove_null_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Remove columns that have more nulls than threshold in %.
    :parameter df: DataFrame to convert
    :parameter threshold: percentage of acceptable non-null values in column
    """
    if 0 <= threshold <= 1:
        proportion_nonnull = df.count()/len(df)
        df_ret = df.loc[:, proportion_nonnull >= threshold]
    else:
        raise ValueError("Invalid threshold!")
    return df_ret


def get_categorical_cols(df: pd.DataFrame, threshold: float):
    """
    Takes a pandas DataFrame as input and returns a list of the categorical columns in that DataFrame.
    """
    categorical_cols = []
    for col in df.columns:
        # Check if the column data type is object
        if df[col].dtype in ['object', 'bool']:
            # Check if the number of unique values is less than or equal to threshold of the total number of values
            if df[col].nunique() <= threshold * df[col].count():
                # Append the column name to the list of categorical columns
                categorical_cols.append(col)

    # Return the list of categorical columns
    return categorical_cols

def get_numerical_cols(df: pd.DataFrame):
    """
    Takes a pandas DataFrame as input and returns a list of the numerical columns in that DataFrame.
    """
    numerical_cols = []
    for col in df.columns:
        # Check if the column data type is numerical (either int or float)
        if df[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)

    # Return the list of numerical columns
    return numerical_cols

def clear_text_data(txt: str):
    lemmatizer = WordNetLemmatizer()
    # txt = txt.lower()
    txt = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", txt)
    stop = stopwords.words('english')
    # remove also nouns because they are not carrying emotions
    txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split() if word not in stop])
    return txt

def main(dataset_name: str):

    if dataset_name == "amazon":
        params_yaml = load_params("params.yaml")["amazon"]
    elif dataset_name == "polarity":
        params_yaml = load_params("params.yaml")["polarity"]
    else: raise ValueError

    test_data_path = params_yaml["split_data"]["test_directory_data"] + f"/TestData{dataset_name}.json"
    train_data_path = params_yaml["split_data"]["train_directory_data"] + f"/TrainData{dataset_name}.json"

    text_columns = params_yaml["feature_extraction"]["text_columns"]
    exclude_columns = params_yaml["feature_extraction"]["exclude_columns"]
    y_name = params_yaml["feature_extraction"]["y_name"]

    df_test = pd.read_json(f"{test_data_path}")
    df_train = pd.read_json(f"{train_data_path}")

    #    Usuń kolumny wszystkie opócz 'overall'
    df_test_removed = remove_features(df_test, f"{y_name}")
    df_test_withNone = add_nan_column(df_test_removed, "ColNone")

    df_train_removed = remove_features(df_train, f"{y_name}")
    df_train_withNone = add_nan_column(df_train_removed, "ColNone")

    df_test_withNone.to_json(
        f'{params_yaml["feature_extraction"]["test_feature_extracted"]}/TestFeaturesExtracted{dataset_name}.json'
    )
    df_train_withNone.to_json(
        f'{params_yaml["feature_extraction"]["train_feature_extracted"]}/TrainFeaturesExtracted{dataset_name}.json'
    )

    # ZADANIE 1 Z LISTY: 4
    # Wczytaj dane i wyrunuj skrypt żeby zobaczyć jakie są tam dane
    # skonstruuj cechy na podstawie kolumn dla zbiorów testowego i treningowego
    # doklej te cechy do danych po preprocessingu i zapisz
    # pierwsza cecha: czy kupione w okolicach swiat Bozego Narodzenia
    if dataset_name == "amazon":
        df_isbough4xmas_test = df_test["reviewTime"].apply(
            lambda x: 11 <= int(x[:3]) or int(x[:3]) <= 1
        )
        df_isbough4xmas_train = df_train["reviewTime"].apply(
            lambda x: 11 <= int(x[:3]) or int(x[:3]) <= 1
        )

        # druga cecha, dlugośc recenzji
        df_review_len_test = df_test["reviewText"].apply(
            lambda x: len(x) if x is not None else 0
        )
        df_review_len_train = df_train["reviewText"].apply(
            lambda x: len(x) if x is not None else 0
        )

        # Ilość dni, które upłynęły między datą recenzji a najnowszą datą recenzji
        df_review_time_test = pd.to_datetime(df_test["reviewTime"])
        df_review_time_train = pd.to_datetime(df_train["reviewTime"])
        df_freshness_test = (df_review_time_test.max() - df_review_time_test).dt.days
        df_freshness_train = (df_review_time_train.max() - df_review_time_train).dt.days

        #     Teraz dodaję te cechy do danych jako nowe kolumny.
        df_test["isBoughtForChristmas"] = df_isbough4xmas_test
        df_train["isBoughtForChristmas"] = df_isbough4xmas_train

        df_test["reviewLength"] = df_review_len_test
        df_train["reviewLength"] = df_review_len_train

        df_test["freshness"] = df_freshness_test
        df_train["freshness"] = df_freshness_train

    df_test.to_json(f'{params_yaml["feature_extraction"]["test_features_appended"]}/TestFeaturesAppended{dataset_name}.json')
    df_train.to_json(f'{params_yaml["feature_extraction"]["train_features_appended"]}/TrainFeaturesAppended{dataset_name}.json')

    # Czyszczenie danych i przetwarzanie wstępne
    if dataset_name == "amazon":
        df_train["vote"].fillna('0', inplace=True)
        df_test["vote"].fillna('0', inplace=True)

        # Remove all columns that have less non-nans than 90% of all values
        df_train_clear = remove_null_columns(df_train, 0.9)
        df_test_clear = remove_null_columns(df_test, 0.9)
    elif dataset_name == "polarity":
        df_train_clear = df_train
        df_test_clear = df_test
    else: raise ValueError
    # if there are still some missing values replace them with mode
    df_train_clear = df_train_clear.fillna(df_train_clear.mode().iloc[0])
    df_test_clear = df_test_clear.fillna(df_test_clear.mode().iloc[0])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Drop useless columns
    df_train_clear.drop(exclude_columns, axis=1, inplace=True)
    df_test_clear.drop(exclude_columns, axis=1, inplace=True)
    # LAB5 UPDATED
    # Refactoring pod katem ulepszenia pipeline
    categorical_columns = params_yaml["feature_extraction"]["categorical_columns"]
    X_train, y_train = df_train_clear.drop([y_name], axis=1), df_train_clear[y_name]
    X_test, y_test = df_test_clear.drop([y_name], axis=1), df_test_clear[y_name]
    numerical_columns = params_yaml["feature_extraction"]["numerical_columns"]

    # perform cleaning of the text, spacy is a lot slower than previous clear_text_data...
    X_train[text_columns] = X_train[text_columns].applymap(clear_text_data)
    X_test[text_columns] = X_test[text_columns].applymap(clear_text_data)

    # concat text columns into one, make the name of column == first of text columns lst
    if dataset_name == "amazon":
        X_train[f"{text_columns[0]}"] = X_train[text_columns].apply(lambda x: ' '.join(x), axis=1)
        X_test[f"{text_columns[0]}"] = X_test[text_columns].apply(lambda x: ' '.join(x), axis=1)
        # Drop columns, one with text is enough
        X_train.drop(text_columns[1:], axis=1, inplace=True)
        X_test.drop(text_columns[1:], axis=1, inplace=True)

    pd.concat([X_train, y_train], axis=1).to_json(f'{params_yaml["feature_extraction"]["train_pipeline"]}/TrainPipeline{dataset_name}.json')
    pd.concat([X_test, y_test], axis=1).to_json(f'{params_yaml["feature_extraction"]["test_pipeline"]}/TestPipeline{dataset_name}.json')
    print(pd.concat([X_train, y_train], axis=1).columns)

if __name__ == "__main__":
    typer.run(main)

