from datetime import datetime
import wandb
import torch

from utils.scenario_lib import scenario_lib
from utils.av_policy import RL_brain
from utils.environment import Env
from utils.function import train_av_online, train_av


eval_size = 4096
batch_size = 128
train_size = 128
epochs = 20
episodes = 20
learning_rate = 1e-4
use_wandb = True
sumo_gui = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lib = scenario_lib(path='./scenario_lib/', 
                   npy_path='./all_data.npy')
env = Env(max_bv_num=lib.max_bv_num, 
          cfg_sumo='./config/lane.sumocfg', 
          gui=sumo_gui)
av_model = RL_brain(env, capacity=train_size*lib.max_timestep, device=device, 
                    batch_size=batch_size, lr=learning_rate)

if use_wandb:
    wandb_logger = wandb.init(
        project='CL for Autonomous Vehicle Training and Testing', 
        name=datetime.now().strftime('%Y%m%d-%H%M')+'-test_av', # for example: '20230509-1544-test_av'
        reinit=True, 
        )
else:
    wandb_logger = None

index = lib.sample(size=train_size)
train_scenario = lib.data[index]

# av_model = train_av_online(av_model, env, scenarios=train_scenario, 
#                            episodes=episodes, wandb_logger=wandb_logger)
av_model = train_av(av_model, env, scenarios=train_scenario, 
                    epochs=epochs, episodes=episodes, wandb_logger=wandb_logger)

env.close()
