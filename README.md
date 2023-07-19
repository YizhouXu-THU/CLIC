# CL-for-Automous-Vehicle-Training-and-Testing

## Preparation

Before running any Python files in `./run/`, please ensure that you have installed `Python>=3.9` and `SUMO>=1.12.0`, then run the following command on the terminal:

```
cd ~/CL-for-Autonomous-Vehicle-Training-and-Testing	# may need to be modified to your actual path
pip install -r requirements.txt
```

## Run

Run Python file directly: 

```
python run/main/main.py
```

Run in the background (can be monitored in real-time on the Weights & Biases webpage):

```
nohup python run/main/main.py
```
