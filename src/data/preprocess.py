#testing.
import os
import argparse
import wandb
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

def normalize_datasets(train, val, test):
    """
    Standardizes features by removing the mean and scaling to unit variance.

    Parameters:
        train, val, test: Tuples in the form (X, y)

    Returns:
        Normalized versions of (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    # Fit scaler on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply same scaler to validation and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

def preprocess_and_log(steps):

    with wandb.init(project="MLOps-Pycon2023",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    
        processed_data = wandb.Artifact(
            "iris-preprocess", type="dataset",
            description="Preprocessed IRIS dataset",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('iris-raw:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)

steps = {"normalize": True}

preprocess_and_log(steps)
