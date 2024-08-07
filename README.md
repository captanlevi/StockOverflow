### COMP9491

## Autonomous agent for short term stock market trading

This repository includes the code for our project, and includes the following components:

* Dataset assembly
* Gym Environment
* Model Training
* Evaluation
* Sample Notebooks

---

## Installation

Create a new Python virtual environment, and run the following command from the project directory:
```sh
pip install -r requirements.txt
``` 

## Execution

Run the following command to train the Q-learning model on the desired stock:
```sh
python3 run.py --filepath data/5minute/APOLLOHOSP.csv --algo Q_learning
```

Similarly, the command for training PPO is:
```sh
python3 run.py --filepath data/5minute/APOLLOHOSP.csv --algo PPO
```

Additional parameters include `model_save_path` and `iterations` but are optional. 

---

Since the project is intended for a real world stock trading scenario, there is no direct inference service but sample executions are given in the project notebooks here.

