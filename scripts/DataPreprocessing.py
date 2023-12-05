#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import gzip
import json
import os
import typer
import yaml


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def load_params(param_path: str):
    with open(param_path, "r") as f:
        return yaml.safe_load(f)


def df_from_json(path: str) -> pd.DataFrame:
    return pd.read_json(f"{path}")

def main(dataset_name: str):
    if dataset_name == "amazon":
        # load params from params.yaml
        params_yaml = load_params("params.yaml")["amazon"]
        in_data_path = params_yaml["data_source"]["in_data_directory"]
        clean_data_path = params_yaml["data_source"]["clean_data_directory"]
        all_beauty_path = params_yaml["data_source"]["all_beauty_source"]
        amazon_fashion_path = params_yaml["data_source"]["amazon_fashion_source"]
        appliances_path = params_yaml["data_source"]["appliances_source"]
        software_path = params_yaml["data_source"]["software_source"]
        # imread data
        # df_All_Beauty_5 = getDF(f'{in_data_path}/All_Beauty_5.json.gz')
        # df_AMAZON_FASHION_5 = getDF(f'{in_data_path}/AMAZON_FASHION_5.json.gz')
        # df_Appliances_5 = getDF(f'{in_data_path}/Appliances_5.json.gz')
        # df_Software_5 = getDF(f'{in_data_path}/Software_5.json.gz')
        df_All_Beauty_5 = getDF(f"{all_beauty_path}")
        df_AMAZON_FASHION_5 = getDF(f"{amazon_fashion_path}")
        df_Appliances_5 = getDF(f"{appliances_path}")
        df_Software_5 = getDF(f"{software_path}")

        df_lst = [df_All_Beauty_5, df_AMAZON_FASHION_5, df_Appliances_5, df_Software_5]
        # Prepare shapes for checking if concatenation is done correctly
        shapes = [df.shape for df in df_lst]
        total_shape = tuple(sum(i) for i in zip(*shapes))

        # Concatenate all dataframes together
        df_concat = pd.concat(df_lst, ignore_index=True)
        # Add new column
        df_concat = df_concat.assign(
            category=[f"All_Beauty_5"] * len(df_All_Beauty_5)
            + [f"AMAZON_FASHION_5"] * len(df_AMAZON_FASHION_5)
            + [f"Appliances_5"] * len(df_Appliances_5)
            + [f"Software_5"] * len(df_Software_5)
        )
        # Reset index, remove multi-index
        df_concat = df_concat.reset_index(drop=True)

        # Solve the problem of nested attribute 'style'
        df_style = pd.json_normalize(df_concat["style"])
        df_concat = pd.concat([df_concat, df_style], axis=1)

        df_concat.drop("style", inplace=True, axis=1)

        print(df_concat.head())
        print(f"Overall shape of df: {df_concat.shape}")
        if "style" in df_concat.columns:
            print(df_concat["style"])
        # Run checks
        assert "category" in df_concat.columns
        # assert df_concat.shape[1] == 12
        assert df_concat.shape[0] == total_shape[0]

        # Save data to json file
        df_concat.to_json(f"{clean_data_path}/Preprocessed_Review_Data_{dataset_name}.json")

    elif dataset_name == "polarity":
        params_yaml = load_params("params.yaml")["polarity"]
        in_data_path = params_yaml["data_source"]["in_data_directory"]
        clean_data_path = params_yaml["data_source"]["clean_data_directory"]
        # imread positive and negative files
        df_pos = pd.read_csv(f"{in_data_path}/rt-polarity.pos", delimiter="\t", header=None, names=["reviewText"],
                             encoding="Windows-1252")
        df_neg = pd.read_csv(f"{in_data_path}/rt-polarity.neg", delimiter="\t", header=None, names=["reviewText"],
                             encoding="Windows-1252")
        y_name = params_yaml["feature_extraction"]["y_name"]
        # define negative and positive sentiment
        df_neg[f"{y_name}"] = 0
        df_pos[f"{y_name}"] = 1
        # concat dfs
        df_concat = pd.concat([df_neg, df_pos], axis=0)
        # save to file
        df_concat.to_json(f"{clean_data_path}/Preprocessed_Review_Data_{dataset_name}.json", orient="records")
    else: raise ValueError

if __name__ == "__main__":
    typer.run(main)
