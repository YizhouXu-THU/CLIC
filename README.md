# CLIC

## [Paper](https://arxiv.org/abs/2309.14209 "Arxiv") | [Code](https://github.com/YizhouXu-THU/CLIC "Github") | [Page](https://sites.google.com/view/icra2024clic "Project page")

This is the repository of paper ***Continual Driving Policy Optimization with Closed-Loop Individualized Curricula*** **(CLIC)**. We develop a continual driving policy optimization framework which we factorize into a set of standardized sub-modules for flexible implementation choices: **AV Evaluation**, **Scenario Selection**, and **AV Training**. CLIC frames AV Evaluation as a collision prediction task, where it estimates the chance of AV failures in these scenarios at each iteration. Subsequently, by re-sampling from historical scenarios based on these failure probabilities, CLIC tailors individualized curricula for downstream training, aligning them with the evaluated capability of AV. Accordingly, CLIC not only maximizes the utilization of the vast pre-collected scenario library for closed-loop driving policy optimization but also facilitates AV improvement by individualizing its training with more challenging cases out of those poorly organized scenarios. This repository provides the codebase on which we benchmark CLIC and baselines in the scenario library generated by (Re)`<sup>`2 `</sup>`H2O ([Paper](https://arxiv.org/abs/2302.13726 "Arxiv") | [Code](https://github.com/Kun-k/Re_2_H2O "Github")).

![CLIC](CLIC.png)

## Installation and Setups

Please first ensure that you have installed `SUMO>=1.12.0` following the [SUMO Documentation](https://sumo.dlr.de/docs/Installing/index.html).

Then run the following command to clone this repository:

```bash
git clone https://github.com/YizhouXu-THU/CLIC.git
cd CLIC
```

You can create a virtual environment for this repository (**Optional**):

```bash
conda env create -n CLIC python=3.10.13
conda activate CLIC
```

Then install the dependencies which are necessary (with `Python>=3.9`):

```bash
pip install -r requirements.txt
```

You should additionally install `torch` following the [PyTorch official website](https://pytorch.org/) based on your system type and CUDA version (or use CPU only). For example:

```bash
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Data

In `./data/` of this repo, only example data is uploaded for testing the code process. If you need complete data to reproduce experimental results, please choose any one to download from [Google Cloud Drive](https://drive.google.com/drive/folders/1LaMbpZL7RwVNiqCizHRe7qBwRJ1fwIds?usp=sharing) and rename the unzipped folder as `data`.

The complete file structure under `./data/` is one of the following two types:

(1) NPZ file `data_npz.zip` (**recommend**):

```
./data
  ├─all.npz
  ├─accident.npz
  └─example.npz
```

(2) CSV file `data_csv.zip`:

```
./data
  ├─all
  │  ├─bv=1
  |  |  └─*.csv
  │  ├─bv=2
  │  ├─bv=3
  │  └─bv=4
  ├─accident
  │  └─...
  └─example
     └─...
```

`./data/all/` categorizes and stores all the scenarios used in this work based on the number of BVs, and the scenarios in which BVs collide with each other during generation have been eliminated; `./data/accident/` extracts scenarios in which AV had an accident during generation (this does not mean that AV will also have an accident here!). `./data/*.npz` are the files stored in `.npz` format for the data in the corresponding folder, generated by running `./run/generate_npy.py`. It can significantly speed up data reading and reduce storage space, but is relatively inconvenient to observe specific values.

An example of scenario data is as follows (`./data/all/bv=2/45085.csv`):

| time_step |  veh_id  |  x_pos  |  y_pos  |  speed  |    yaw    |
| :-------: | :------: | :------: | :------: | :------: | :-------: |
| 0.00E+00 | 0.00E+00 | 2.41E+01 | 4.58E+00 | 3.13E+01 | 1.28E-03 |
| 0.00E+00 | 1.00E+00 | 2.12E+01 | 8.39E+00 | 2.59E+01 | 2.31E-03 |
| 0.00E+00 | 2.00E+00 | 5.00E+00 | 4.67E+00 | 3.07E+01 | 2.60E-03 |
| 1.00E+00 | 1.00E+00 | 2.23E+01 | 8.35E+00 | 2.62E+01 | -3.96E-02 |
| 1.00E+00 | 2.00E+00 | 6.24E+00 | 4.72E+00 | 3.09E+01 | 4.42E-02 |
| 2.00E+00 | 1.00E+00 | 2.33E+01 | 8.26E+00 | 2.64E+01 | -8.12E-02 |
| 2.00E+00 | 2.00E+00 | 7.48E+00 | 4.83E+00 | 3.11E+01 | 8.58E-02 |
| 3.00E+00 | 1.00E+00 | 2.44E+01 | 8.13E+00 | 2.66E+01 | -1.23E-01 |
| 3.00E+00 | 2.00E+00 | 8.71E+00 | 4.99E+00 | 3.12E+01 | 1.27E-01 |
| 4.00E+00 | 1.00E+00 | 2.54E+01 | 7.96E+00 | 2.69E+01 | -1.64E-01 |
| 4.00E+00 | 2.00E+00 | 9.95E+00 | 5.20E+00 | 3.13E+01 | 1.68E-01 |

Here the first row records the initial state of AV (`veh_id=0`), and each subsequent row records the state of each BV (`veh_id=1,2`) at each timestep, while the AV state at subsequent timesteps is obtained by rollout. The unit of speed is $\text{m/s}$, and the unit of yaw is $\text{rad}$.

The NPZ file can be read using the `numpy.load()` function to obtain a dict, which contains 4 pieces of data:

* `'scenario'`: A 2-d numpy array containing all scenario data after filling with 0 (referring to the `scenario_lib.fill_zero()` function in `./utils/scenario_lib.py`), where each row is one flattened scenario.
* `'label'`: A 1-d numpy array with a length equal to the number of scenarios, containing labels indicating whether an accident occurred in AV during a certain experiment, which can be used as a ground truth to train the predictor and test its performance qiuckly.
* `'type_count'`: A dict that calculates the number of scenarios corresponding to different number of BV, where key is the number of BV in the scenario, and value is the corresponding number of scenarios.
* `'max_bv_num'`: The maximum number of BV in all scenarios.

## Run Experiments

Run the main training and testing experiment directly:

```bash
python run/main/main.py
```

And you can also try baselines by running `./run/main/*.py`, or modify some experimental settings and hyperparameters in these files.

If you want to observe the real-time motion states of the vehicles through SUMO GUI, set parameter `sumo_gui=True`, and ensure your device has already installed SUMO GUI.

If you want to save the AV and predictor model for each iteration as checkpoints during the training process, set parameter `save_model=True`.

## Visualization of Learning Curves

You can resort to [Weights &amp; Biases](https://wandb.ai/site) to login your personal account with your wandb API key.

```bash
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```

and set parameter `use_wandb=True` in `./run/main/xxx.py` files to turn on the online syncronization.

## Citation

If you are using CLIC framework or code for your project development, please cite the following paper:

```
@inproceedings{niu2024continual,
  title={Continual Driving Policy Optimization with Closed-Loop Individualized Curricula},
  author={Niu, Haoyi and Xu, Yizhou and Jiang, Xingjian and Hu, Jianming},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6850--6857},
  year={2024},
  organization={IEEE}
}
```
