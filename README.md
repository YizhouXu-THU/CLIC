# CLIC

This is the repo of Continual Driving Policy Optimization with **C**losed-**L**oop **I**ndividualized **C**urricula (CLIC) ([https://arxiv.org/abs/2309.14209](https://arxiv.org/abs/2309.14209)). 

![CLIC](figure/algorithm.png)

## Installation and Setups

Please ensure that you have installed `Python>=3.9` and `SUMO>=1.12.0`, then run the command to install the dependencies: 

```
pip install -r requirements.txt
```

Add this repo directory to your `PYTHONPATH` environment variable: 

```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Experiments

Run the main training and testing experiment directly: 

```
python run/main/main.py
```

And you can also try baselines by running `run/main/*.py`. 

## Visualization of Learning Curves

You can resort to [wandb](https://wandb.ai/site) to login your personal account with your wandb API key. 

```
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```

and set parameter `use_wandb=True` to turn on the online syncronization. 

## Citation

If you are using CLIC framework or code for your project development, please cite the following paper: 

```
@article{niu2023continual,
  title={Continual Driving Policy Optimization with Closed-Loop Individualized Curricula},
  author={Niu, Haoyi and Xu, Yizhou and Jiang, Xingjian and Hu, Jianming},
  journal={arXiv preprint arXiv:2309.14209},
  year={2023}
}
```
