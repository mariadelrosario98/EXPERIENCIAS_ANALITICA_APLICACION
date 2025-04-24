from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
import pandas as pd 
import argparse
import wandb 
import numpy as np
from io import StringIO

#
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )

    # Calculate adjusted validation size relative to temp set
    val_ratio = val_size / (1 - train_size)

    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_and_log():
    # 🚀 Start a W&B run
    with wandb.init(
        project="MLOps-Pycon2023-2",
        name=f"Load Raw Data ExecId-{args.IdExecution}",
        job_type="load-data"
    ) as run:

        # Load datasets
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load()

        # Convert to pandas DataFrames
        train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        train_df["target"] = y_train

        val_df = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
        val_df["target"] = y_val

        test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        test_df["target"] = y_test

        datasets = [train_df, val_df, test_df]
        names = ["training", "validation", "test"]

        # 🏺 Create the artifact
        raw_data = wandb.Artifact(
            name="iris-raw",
            type="dataset",
            description="Iris dataset split into train/val/test",
            metadata={"source": "sklearn.datasets.load_iris",
                      "sizes": [len(df) for df in datasets]}
        )

        # 📝 Save each DataFrame as CSV to the artifact
        for name, df in zip(names, datasets):
            with raw_data.new_file(f"{name}.csv", mode="w") as f:
                df.to_csv(f, index=False)

        # 📦 Log the artifact to W&B
        run.log_artifact(raw_data)

# testing
load_and_log()
