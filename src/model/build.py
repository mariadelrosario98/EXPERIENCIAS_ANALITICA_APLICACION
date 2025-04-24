# Import the model class from the main file.
from sklearn.linear_model import LinearRegression
import os
import argparse
import wandb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")

# Data parameters testing
input_shape = 4

# Define the model filename before saving
model_filename = "linear_regression_model.pkl"

def build_model_and_log(config, model, model_name="Linear Regression", model_description="Simple Linear"):
    with wandb.init(project="MLOps-Pycon2023", 
                    name=f"initialize Model ExecId-{args.IdExecution}", 
                    job_type="initialize-model", config=config) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))
        
        # Save the trained model
        with open(f"./model/{model_filename}", "wb") as f:
            pickle.dump(model, f)

        # Add the model to the artifact
        model_artifact.add_file(f"./model/{model_filename}")

        # Corrected line: use the defined model_filename
        wandb.save(f"./model/{model_filename}")

        run.log_artifact(model_artifact)


# Model configuration
model_config = {"input_shape": input_shape}

model = LinearRegression()

# Log the model and configuration
build_model_and_log(model_config, model, "linear", "Simple Linear Regression Model")