#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from DataPreprocessing import load_params

def main(dataset_name:str):
    if dataset_name == 'amazon':
        params_yaml = load_params("params.yaml")['amazon']
    elif dataset_name == 'polarity':
        params_yaml = load_params("params.yaml")['polarity']
    else: raise ValueError

    clean_data_path = params_yaml["data_source"]["clean_data_directory"] + f"/Preprocessed_Review_Data_{dataset_name}.json"
    test_data_path = params_yaml["split_data"]["test_directory_data"] + f"/TestData{dataset_name}.json"
    train_data_path = params_yaml["split_data"]["train_directory_data"]+ f"/TrainData{dataset_name}.json"
    df_clean_data = pd.read_json(f"{clean_data_path}")
    y_name = params_yaml["feature_extraction"]["y_name"]
    print(df_clean_data.head())
    # Split data into two parts: X and y
    X = df_clean_data.loc[:, df_clean_data.columns != f"{y_name}"]
    y = df_clean_data.loc[:, df_clean_data.columns == f"{y_name}"]

    # while splitting, it is needed to shuffle after concatenating
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params_yaml["split_data"]["test_size"],
        random_state=42,
        shuffle=True,
        stratify=y,
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # zapisanie dodatkowo wspólnie setów
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] == y_train.shape[0]

    df_train.to_json(f"{train_data_path}")
    df_test.to_json(f"{test_data_path}")


if __name__ == "__main__":
    typer.run(main)
