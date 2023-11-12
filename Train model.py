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
