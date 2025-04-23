from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
import pandas as pd 
import argparse
import wandb 


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
    # 🚀 start a run, with a type to label it and a project it can call home.
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "iris-raw", type="dataset",
            description="raw iris dataset, split into train/val/test",
            metadata={"source": "sklearn.datasets.load_iris",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file) # type: ignore

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
