import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf

print(f"Found TF-DF {tfdf.__version__}")

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
serving_df = pd.read_csv("/kaggle/input/titanic/test.csv")

train_df.head(10)
def custom_normalize_name(x):
    return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
def custom_ticket_number(x):
    return x.split(" ")[-1]
        
def custom_ticket_item(x):
    items = x.split(" ")
    if len(items) == 1:
        return "NONE"
    return "_".join(items[0:-1])
def preprocess_df(df):
    df = df.copy()
    df["Name"] = df["Name"].apply(custom_normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(custom_ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(custom_ticket_item)   
    return df
preprocessed_train_df = preprocess_df(train_df)
preprocessed_serving_df = preprocess_df(serving_df)

preprocessed_train_df.head(5)

input_features = list(preprocessed_train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")
#input_features.remove("Ticket_number")

print(f"Input features: {input_features}")
def tokenize_names(features, labels=None):
    """Divite the names into tokens. TF-DF can consume text tokens natively."""
    features["Name"] =  tf.strings.split(features["Name"])
    return features, labels

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df,label="Survived").map(tokenize_names)
serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)

def train_and_evaluate_model(train_ds, input_features):
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=0,  # Very few logs
        features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
        exclude_non_specified_features=True,  # Only use the features in "input_features"
        random_seed=1234,
    )
