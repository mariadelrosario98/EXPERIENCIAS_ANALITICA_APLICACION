import pandas as pd
import os
import argparse
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"


def read_csv_data(data_dir, split):
    """
    Reads a CSV file and returns X, y.
    """
    filepath = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(filepath)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    return X, y


def train_model(X_train, y_train, config):
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=config["max_iter"],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return acc, f1, precision, recall, cm, report



def train_and_evaluate(config, experiment_id='99'):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Train-Eval LogisticRegression ExecId-{args.IdExecution} ExperimentId-{experiment_id}",
        job_type="train-eval", config=config) as run:

        data = run.use_artifact('iris-preprocessed:latest')
        data_dir = data_artifact.download()

        X_train, y_train = read_csv_data(data_dir, "train")
        X_val, y_val = read_csv_data(data_dir, "validation")
        X_test, y_test = read_csv_data(data_dir, "test")

        # Train
        model = train_model(X_train, y_train, config)

        # Evaluate on validation
        val_acc, val_f1, val_precision, val_recall, _, _ = evaluate_model(model, X_val, y_val)
        print(f"Validation Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Evaluate on test
        test_acc, test_f1, test_precision, test_recall, confusion_mtx, full_report = evaluate_model(model, X_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")

        # Log metrics
        wandb.log({
            "validation/accuracy": val_acc,
            "validation/f1": val_f1,
            "validation/precision": val_precision,
            "validation/recall": val_recall,
            "test/accuracy": test_acc,
            "test/f1": test_f1,
            "test/precision": test_precision,
            "test/recall": test_recall,
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=model.predict(X_test),
                y_true=y_test,
                class_names=[str(i) for i in sorted(set(y_test))]
            )
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if not args.IdExecution:
        args.IdExecution = "testing-console"

    # You can tune these parameters
    iterations = [100, 200, 300]
    
    for id, max_iter in enumerate(iterations):
        config = {
            "max_iter": max_iter
        }
        train_and_evaluate(config, experiment_id=id)
