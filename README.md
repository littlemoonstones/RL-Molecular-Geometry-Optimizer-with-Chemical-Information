# Getting Started

A guide on how to set up the project on a local machine. This section should include steps to install any dependencies required to run the project.

## Installation

```bash
# create a new conda environment
conda env create -f environment.yml
conda activate rl-opt

# install the packages
pip install -r requirements.txt
```

## Brief Folder Structure
Explain the organization of this project's folders and their contents:
```text
.
├── Evaluation
│   ├── compare.py
│   ├── pysisyphus
│   ├── test-configs
│   └── worker.py
└── Train
    ├── configs
    ├── pysisyphus
    └── run.py
```

## Train
- `configs`: Contains the configuration files for training the model.
- `pysisyphus`: this package is modified from the original [pysisyphus](https://github.com/eljost/pysisyphus). `Optimizer.py` is modified to the generator. It is also added `RDKit.py` as a new calcaulator.
- `run.py`: This file is the entry point for starting the model training process.

### Table Configs
In `configs`
| Config File | Description |
| ----------- | ----------- |
| `cart1/v4.ini` | It corresponds to `Group A`. |
| `many5103/v3.ini` | It corresponds to `Group B`. |
| `many510/v3.ini` | It corresponds to `Group C`. |
| `schnet1/v11.ini` | It corresponds to `Group D`. |

## Evaluation
- `test-configs`: you can determine the dataset, the type of coordinates(e.g. cartesian(`cart`) or internal(`redund`) coordinates), the type of calculator(e.g. `mmff` or `psi4`) and the type of optimizer(e.g. `bfgs` or `rl`).
- `worker.py`: This file is to collect the data(`.json`) which the optimizer optimize the molecular geometry.
- `compare.py`: This file is to compare the performance of different optimizers after collecting the data.


# Usage
## Training

### Run the training script

```bash
cd Train
python run.py \
    --name NAME \ 
    --version VERSION \
    --seed SEED
```

#### For example
We select the configuration file at `many510/v3.ini` and set the name as `many510` and the version as `3`. The seed is set as `283484033`. The command is as follows:
```bash
cd Train
python run.py \
    --name many510 \ 
    --version 3 \
    --seed 283484033
```

> After training, the model will be stored in `Train/saves/NAME/VERSION/`. You need to copy or move the model to the `Evaluation` folder if you want to evaluate the model.

### Fine-tune 
```bash
cd Train
python runWithPretrain.py \
    --name many510 \
    --version 3 \
    -r 708700354 \
    --pre_trained saves/RL-many510/v3/seed-283484033/best_model.pt \
```

## Evaluation
Before we compare the performance of different optimizers, we need to collect the data(`.json`) which the optimizer optimize the molecular geometry. The data will be stored in `Result-GROUP_NAME/OPTIMIZER_NAME/`. 

To collect the data, we can run the evaluation script first.
```bash
python worker.py --name GROUP_NAME --opt_key OPTIMIZER_NAME -p
```

### For example
#### BFGS
In our setup, we have chosen BFGS (denoted as `bfgs`) as our optimizer. To utilize a dataset with perturbations, add the `-p` flag. If you prefer to use a dataset without perturbations, simply omit the `-p` flag. The script fetches its settings from the configuration file located at `test-configs/bfgs.yml`, which specifies which dataset to use.

The command to execute the script is:
```bash
python worker.py --name group-C --opt_key bfgs -p
```
#### RL model
To run the dataset without perturbations, use the following command. The script fetches its settings from the configuration file located at `test-configs/rl.yml`, which specifies which dataset to employ.

The command is as follows:
```bash
python worker.py --name group-C --opt_key rl
```

#### Comparison
After collecting the data, we can compare the performance of different optimizers by running the following script.
```bash
python compare.py
```

## Evaluation with e-Baker dataset
```bash
cd e-Baker-Evaluation
python main.py
```