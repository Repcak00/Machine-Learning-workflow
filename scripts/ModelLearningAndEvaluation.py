#!/usr/bin/python
# -*- coding: utf-8 -*-
import json

import sklearn
import typer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, \
    HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from DataPreprocessing import df_from_json, load_params
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.dummy import DummyClassifier
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
import spacy

def plot_fig(cnf_matrix, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    sns.heatmap(cnf_matrix, ax=ax)
    mlflow.log_figure(fig, f"{title}_confusion_matrix.jpg")


nlp = spacy.load("en_core_web_md", exclude=["ner", "senter"])
def clear_text_data_spacy(text: str):
    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and token.pos_ in ["ADJ", "ADV", "VERB", "NOUN"]]
    return ' '.join(words)

def main(dataset_name: str):
    if dataset_name == "amazon":
        params_yaml = load_params("params.yaml")["amazon"]
    elif dataset_name == "polarity":
        params_yaml = load_params("params.yaml")["polarity"]
    else: raise ValueError

    df_train = df_from_json(
        f'{params_yaml["feature_extraction"]["train_feature_extracted"]}/TrainFeaturesExtracted{dataset_name}.json'
    )
    df_test = df_from_json(
        f'{params_yaml["feature_extraction"]["test_feature_extracted"]}/TestFeaturesExtracted{dataset_name}.json'
    )
    y_name = params_yaml["feature_extraction"]["y_name"]
    text_column = params_yaml["feature_extraction"]["text_columns"][0]
    categorical_columns = params_yaml["feature_extraction"]["categorical_columns"]
    numerical_columns = params_yaml["feature_extraction"]["numerical_columns"]

    X_train = df_train["ColNone"]
    y_train = df_train[f"{y_name}"]

    X_test = df_test["ColNone"]
    y_test = df_test[f"{y_name}"]

    # Stworzenie klasyfikatora

    dummy_clf = DummyClassifier(strategy=f"{params_yaml['model_training']['strategy']}")
    # Dummy clf ignoruje klasę X, ale i tak dam ją do .fit()
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    # Używam 'macro' żeby policzyło pozytywne i negatywne globalnie
    score_f1 = f1_score(y_test, y_pred, average="macro")
    metrics = {"f1_score_dummy": score_f1}
    # LAB5
    # reformatting with pipeline
    df_train = df_from_json(
        f'{params_yaml["feature_extraction"]["train_features_appended"]}/TrainPipeline{dataset_name}.json'
    )
    df_test = df_from_json(
        f'{params_yaml["feature_extraction"]["test_features_appended"]}/TestPipeline{dataset_name}.json'
    )

    scoring = params_yaml["model_training"]["scoring"]
    X_train, y_train = df_train.drop([y_name], axis=1), df_train[y_name]
    X_test, y_test = df_test.drop([y_name], axis=1), df_test[y_name]
    if dataset_name == "amazon":
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorize', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                ('standarize', StandardScaler(), numerical_columns),
                ('vectorize', TfidfVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"]), text_column)
            ]
        )
    elif dataset_name == "polarity":
        preprocessor = ColumnTransformer(
            transformers=[
                ('vectorize', TfidfVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"]),
                 text_column)
            ]
        )
    else: raise ValueError
    # only text column
    X_text = X_train[text_column]
    pipeline_svm_text = Pipeline(steps=[('vectorize', TfidfVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"])),
                        ('svm',SVC(C=params_yaml["model_training"]["svm_C"], kernel=params_yaml["model_training"]["svm_kernel"], gamma=params_yaml["model_training"]["svm_gamma"]))])
    pipeline_rf_text = Pipeline(steps=[('vectorize', TfidfVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"])),
                        ('rf',RandomForestClassifier(n_estimators=params_yaml["model_training"]["rf_n_estimators"],
                                 criterion=params_yaml["model_training"]["rf_criterion"],
                                    max_depth=params_yaml["model_training"]["rf_max_depth"]))])
    scores = cross_validate(pipeline_svm_text, X_text, y_train, scoring=scoring, cv=3, n_jobs=-1)
    print("\nCross validation analysis for text only, SVM:")
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
        metrics[f"SVM_text_model_{score}"] = scores[f'test_{score}'].mean()

    scores = cross_validate(pipeline_rf_text, X_text, y_train, scoring=scoring, cv=3, n_jobs=-1)
    print("\nCross validation analysis for text only, RandomForest:")
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
        metrics[f"randomForest_text_model_{score}"] = scores[f'test_{score}'].mean()

    if dataset_name == "amazon":
        #other than text columns
        X_other = X_train.drop([text_column], axis=1)

        pipeline_svm_other = Pipeline(steps=[
                            ('transform_col', ColumnTransformer(transformers=[
                            ('categorize', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                            ('standarize', StandardScaler(), numerical_columns)])),
                            ('svm',SVC(C=params_yaml["model_training"]["svm_C"], kernel=params_yaml["model_training"]["svm_kernel"], gamma=params_yaml["model_training"]["svm_gamma"]))])
        pipeline_rf_other = Pipeline(steps=[
                            ('transform_col', ColumnTransformer(transformers=[
                            ('categorize', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                            ('standarize', StandardScaler(), numerical_columns)])),
                            ('rf',RandomForestClassifier(n_estimators=params_yaml["model_training"]["rf_n_estimators"],
                                     criterion=params_yaml["model_training"]["rf_criterion"],
                                        max_depth=params_yaml["model_training"]["rf_max_depth"]))])
        scores = cross_validate(pipeline_svm_other, X_other, y_train, scoring=scoring, cv=3, n_jobs=-1)
        print("\nCross validation analysis for other than text, SVM:")
        for score in scoring:
            print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
            metrics[f"SVM_non_text_model_{score}"] = scores[f'test_{score}'].mean()

        scores = cross_validate(pipeline_rf_other, X_other, y_train, scoring=scoring, cv=3, n_jobs=-1)
        print("\nCross validation analysis for other than text, RandomForest:")
        for score in scoring:
            print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
            metrics[f"randomForest_non_text_model_{score}"] = scores[f'test_{score}'].mean()

    # all columns
    text_column = params_yaml["feature_extraction"]["text_columns"][0]
    pipeline_svm_all = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('svm', SVC(C=params_yaml["model_training"]["svm_C"],
                                               kernel=params_yaml["model_training"]["svm_kernel"],
                                               gamma=params_yaml["model_training"]["svm_gamma"]))])

    pipeline_rf_all = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('rf', RandomForestClassifier(n_estimators=params_yaml["model_training"]["rf_n_estimators"],
                                                          criterion=params_yaml["model_training"]["rf_criterion"],
                                                          max_depth=params_yaml["model_training"]["rf_max_depth"]))])
    print("\nCross validation analysis for all columns, SVM:")
    scores = cross_validate(pipeline_svm_all, X_train, y_train, scoring=scoring, cv=3, error_score='raise', n_jobs=-1)
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
        metrics[f"SVM_all_data_model_{score}"] = scores[f'test_{score}'].mean()

    print("\nCross validation analysis for all columns, Random Forest:")
    scores = cross_validate(pipeline_rf_all, X_train, y_train, scoring=scoring, cv=4, error_score='raise', n_jobs=-1)
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
        metrics[f"randomForest_all_data_model_{score}"] = scores[f'test_{score}'].mean()

    # dummy clf
    print("\ndummy")
    dummy_clf = DummyClassifier(strategy=f"{params_yaml['model_training']['strategy']}")
    scores = cross_validate(dummy_clf, X_train, y_train, scoring=scoring, cv=3, n_jobs=-1)
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
        metrics[f"dummy_clf_all_data_model_{score}"] = scores[f'test_{score}'].mean()

    # performing feature engineering, dropping features. Also remember to remove names from clolumn lists
    X_train.drop(params_yaml["model_training"]["feature_engineering_drop"], axis=1, inplace=True)
    X_test.drop(params_yaml["model_training"]["feature_engineering_drop"], axis=1, inplace=True)

    with open(f"{params_yaml['metrics']['score_metrics']}/metrics{dataset_name}.json", "w") as f:
        json.dump(metrics, f)

    for col in params_yaml["model_training"]["feature_engineering_drop"]:
        if col in text_column:
            text_column.remove(col)
        elif col in categorical_columns:
            categorical_columns.remove(col)
        elif col in numerical_columns:
            numerical_columns.remove(col)

    # Create the final pipeline that includes the preprocessor and the SVM model

    pipeline_svm = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('pca', TruncatedSVD(n_components=params_yaml["model_training"]["pca_components_number"])),
                                   ('svm', SVC(C=params_yaml["model_training"]["svm_C"],
                                               kernel=params_yaml["model_training"]["svm_kernel"],
                                               gamma=params_yaml["model_training"]["svm_gamma"]))])

    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('pca', TruncatedSVD(n_components=params_yaml["model_training"]["pca_components_number"])),
                                  ('rf', RandomForestClassifier(n_estimators=params_yaml["model_training"]["rf_n_estimators"],
                                                          criterion=params_yaml["model_training"]["rf_criterion"],
                                                          max_depth=params_yaml["model_training"]["rf_max_depth"]))])

    print("\nAfter feature engineering\nSVM:")
    scores = cross_validate(pipeline_svm, X_train, y_train, scoring=scoring, cv=4, error_score='raise', n_jobs=-1)
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")
    print("\nRandom Forest:")
    scores = cross_validate(pipeline_rf, X_train, y_train, scoring=scoring, cv=4, error_score='raise', n_jobs=-1)
    for score in scoring:
        print(f"Cross-validation score {score}: {scores[f'test_{score}'].mean()}")

    # test set
    clf_tab = []
    pipeline_rf.fit(X_train, y_train)
    clf_tab.append(pipeline_rf)

    pipeline_svm.fit(X_train, y_train)
    clf_tab.append(pipeline_svm)

    dummy_clf = DummyClassifier(strategy=f"{params_yaml['model_training']['strategy']}")
    # Dummy clf ignoruje klasę X, ale i tak dam ją do .fit()
    dummy_clf.fit(X_train, y_train)
    clf_tab.append(dummy_clf)

    md_metrics = {}
    for clf in clf_tab:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # f1_macro = f1_score(y_test, y_pred, average='macro')
        precision_micro = precision_score(y_test, y_pred, average='micro')
        recall_macro = recall_score(y_test, y_pred, average='micro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        if type(clf) == sklearn.pipeline.Pipeline:
            md_metrics[type(clf.steps[-1][1]).__name__] = {'f1_micro': f1_micro, 'f1_weighted': f1_weighted,
                                                  'accuracy': accuracy, 'precision_micro': precision_micro,
                                                  'recall_macro': recall_macro}
        else:
            md_metrics[clf.__class__.__name__] = {'f1_micro': f1_micro, 'f1_weighted': f1_weighted, 'accuracy': accuracy, 'precision_micro': precision_micro, 'recall_macro': recall_macro}
    result_df = pd.DataFrame.from_dict(md_metrics, orient='index')
    result_df.index.name = 'Model'
    result_table = result_df.to_markdown()
    print(result_table)
    with open(f'{params_yaml["reports"]["markdown_table"]}/metrics_table{dataset_name}.md', 'w') as f:
        f.write(result_table)

    # LAB6
    if dataset_name == "amazon":
        param_svm_hgs = {
            'preprocessor__vectorize__max_features': [800, 1200],
            'pca__n_components': [200, 300],
            'svm__C': [0.1, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': [0.01, 0.9]
        }
        param_rf_gs = {
            'preprocessor__vectorize__max_features': [800, 1200],
            # 'pca__n_components': [200, 300],
            'rf__n_estimators': [120, 180],
            # 'rf__criterion': ["gini", "entropy"],
            # 'rf__max_depth': [120, 200]
        }
        param_rf_rs = {
            'preprocessor__vectorize__max_features': [800, 1200],
            'pca__n_components': [200, 300],
            'rf__n_estimators': [120, 180],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_depth': [120, 200]
        }
        param_rf_hgs = {
            'preprocessor__vectorize__max_features': [800, 1200],
            'pca__n_components': [200, 300],
            'rf__n_estimators': [120, 180],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_depth': [120, 200]
        }
        param_rf_hrs = {
            'preprocessor__vectorize__max_features': [800, 1200],
            'pca__n_components': [200, 300],
            'rf__n_estimators': [120, 180],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_depth': [120, 200]
        }
    elif dataset_name == "polarity":
        param_svm_hgs = {
            'preprocessor__vectorize__max_features': [300, 200],
            'pca__n_components': [15, 20],
            'svm__C': [0.1, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': [0.01, 0.9]
        }
        param_rf_gs = {
            'preprocessor__vectorize__max_features': [300, 200],
            # 'pca__n_components': [20, 25],
            'rf__n_estimators': [120, 180],
            # 'rf__criterion': ["gini", "entropy"],
            # 'rf__max_depth': [120, 200]
        }
        param_rf_rs = {
            'preprocessor__vectorize__max_features': [300, 200],
            'pca__n_components': [20, 25],
            'rf__n_estimators': [120, 180],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_depth': [120, 200]
        }
        param_rf_hgs = {
            'preprocessor__vectorize__max_features': [300, 200],
            'pca__n_components': [20, 25],
            'rf__n_estimators': [120, 180],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_depth': [120, 200]
        }
        param_rf_hrs = {
            'preprocessor__vectorize__max_features': [300, 200],
            'pca__n_components': [15, 20],
            'rf__n_estimators': [120, 180],
            'rf__criterion': ["gini", "entropy"],
            'rf__max_depth': [120, 200]
        }
    else: raise ValueError
    # Gridsearch przeprowadza każdą możliwą kombinację, dlatego bardzo długo trwa. Wykładniczo to rośnie
    grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_rf_gs, cv=2, n_jobs=-1, verbose=True)
    # Liczba stworzonych modeli zależy od parametru n_iter. Parametry dobierane są losowo
    randomized_search_rf = RandomizedSearchCV(pipeline_rf, param_distributions=param_rf_rs, cv=2, n_jobs=-1, n_iter=5, verbose=True, random_state=0)
    # strategia polega na iteracyjnym wybieraniu kandydatów (najlepszych). Kandydaci są próbkowani losowo.
    halving_random_search_rf = HalvingRandomSearchCV(pipeline_rf, param_rf_hrs, cv=2, factor=4, n_jobs=-1, random_state=0, max_resources=200, n_candidates=16)
    # halving grid search dobiera najlepszych kandydatów. Ranking kadydatów jest jednak ustalany na podstawie score'a, a nie randomowo jak w przypadku halving random search
    halving_grid_search_rf = HalvingGridSearchCV(pipeline_rf, param_rf_hgs, cv=2, factor=4, n_jobs=-1)

    # SVM
    halving_grid_search_svm = HalvingGridSearchCV(pipeline_svm, param_svm_hgs, cv=2, factor=4, n_jobs=-1)

    # fitting
    grid_search_rf.fit(X_train, y_train)
    randomized_search_rf.fit(X_train, y_train)  #10 minut
    halving_random_search_rf.fit(X_train, y_train)
    halving_grid_search_rf.fit(X_train, y_train)
    halving_grid_search_svm.fit(X_train, y_train)

    print(f"Grid search RF, best params: {grid_search_rf.best_params_}\nBest score: {grid_search_rf.best_score_}\nTime: {grid_search_rf.cv_results_['mean_fit_time'].sum()}")
    print(f"Randomized Search RF, best params: {randomized_search_rf.best_params_}\nBest score: {randomized_search_rf.best_score_}\nTime: {randomized_search_rf.cv_results_['mean_fit_time'].sum()}")
    print(f"Halving Random Search RF, best params: {halving_random_search_rf.best_params_}\nBest score: {halving_random_search_rf.best_score_}\nIter: {halving_random_search_rf.n_iterations_}")
    print(f"Halving Grid Search RF, best params: {halving_grid_search_rf.best_params_}\nBest score: {halving_grid_search_rf.best_score_}\nIter: {halving_grid_search_rf.n_iterations_}\nTime: {halving_grid_search_rf.cv_results_['mean_fit_time'].sum()}")
    print(f"Halving Grid Search SVM, best params: {halving_grid_search_svm.best_params_}\nBest score: {halving_grid_search_svm.best_score_}\nIter: {halving_grid_search_svm.n_iterations_}\nTime: {halving_grid_search_svm.cv_results_['mean_fit_time'].sum()}")

    times = [grid_search_rf.cv_results_['mean_fit_time'].sum(), randomized_search_rf.cv_results_['mean_fit_time'].sum(),
             halving_grid_search_rf.cv_results_['mean_fit_time'].sum(), halving_random_search_rf.cv_results_['mean_fit_time'].sum()]
    scores = [grid_search_rf.best_score_, randomized_search_rf.best_score_, halving_grid_search_rf.best_score_, halving_random_search_rf.best_score_]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(['GS', 'RS', 'HGS', 'HRS'], times)
    ax[0].set_title("Search time")
    ax[0].set_ylabel("Time (seconds)")
    ax[1].bar(['GS', 'RS', 'HGS', 'HRS'], scores)
    ax[1].set_title("Best score")
    ax[1].set_ylabel("Score")
    ax[0].set_title("Search time")
    ax[1].set_title("Best score")
    fig.savefig(f"{params_yaml['reports']['hiperparameters_tuning']}/hiperparameters_tuning_fig{dataset_name}.png")

    #  dokonaj preprocessingu danych: korzystam z funkcji clear_text_data_spacy
    X_train[text_column] = X_train[text_column].apply(clear_text_data_spacy)
    X_test[text_column] = X_test[text_column].apply(clear_text_data_spacy)
    #  dokonaj wektoryzacji tak przetworzonego tekstu przy pomocy następujących metod: BoW - CountVectorizer, tf-idf, word2vec
    # Ta część i późniejsze wykonane są w notebooku
    if dataset_name == "amazon":
        preprocessor_BoW = ColumnTransformer(
            transformers=[
                ('categorize', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                ('standarize', StandardScaler(), numerical_columns),
                ('vectorize', CountVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"]), text_column)
            ]
        )
        preprocessor_tfidf = ColumnTransformer(
            transformers=[
                ('categorize', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                ('standarize', StandardScaler(), numerical_columns),
                ('vectorize', TfidfVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"]), text_column)
            ]
        )
    elif dataset_name == "polarity":
        preprocessor_BoW = ColumnTransformer(
            transformers=[
                ('vectorize', CountVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"]), text_column)
            ]
        )
        preprocessor_tfidf = ColumnTransformer(
            transformers=[
                ('vectorize', TfidfVectorizer(max_features=params_yaml["model_training"]["vectorizer_max_features"]), text_column)
            ]
        )
    else: raise ValueError
    #
    pipeline_rf_BoW = Pipeline(steps=[('preprocessor_BoW', preprocessor_BoW),
                                  ('pca', TruncatedSVD(n_components=params_yaml["model_training"]["pca_components_number"])),
                                  ('rf', RandomForestClassifier(n_estimators=params_yaml["model_training"]["rf_n_estimators"],
                                                          criterion=params_yaml["model_training"]["rf_criterion"],
                                                          max_depth=params_yaml["model_training"]["rf_max_depth"]))])
    pipeline_rf_tfidf = Pipeline(steps=[('preprocessor_tfidf', preprocessor_tfidf),
                                  ('pca', TruncatedSVD(n_components=params_yaml["model_training"]["pca_components_number"])),
                                  ('rf', RandomForestClassifier(n_estimators=params_yaml["model_training"]["rf_n_estimators"],
                                                          criterion=params_yaml["model_training"]["rf_criterion"],
                                                          max_depth=params_yaml["model_training"]["rf_max_depth"]))])



    pipeline_rf_BoW.fit(X_train, y_train)
    print(f"Vectorization using BoW, f1 weighted: {f1_score(y_test, pipeline_rf_BoW.predict(X_test), average='weighted')}")
    pipeline_rf_tfidf.fit(X_train, y_train)
    print(f"Vectorization using TfIdf, f1 weighted: {f1_score(y_test, pipeline_rf_tfidf.predict(X_test), average='weighted')}")


if __name__ == "__main__":
    typer.run(main)
