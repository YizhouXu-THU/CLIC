import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils.predictor import predictor


class scenario_lib:
    def __init__(self, path='./scenario_lib/', npy_path='./all_data.npy') -> None:
        self.path = path
        self.npy_path = npy_path
        self.max_bv_num = 0                         # initialize with 0
        self.max_timestep = 0                       # initialize with 0
        self.type_count = {}                        # count separately based on the number of bv
        self.data = self.load_data()
        self.scenario_num = self.data.shape[0]
        self.max_dim = self.data.shape[1]
        self.labels = np.ones(self.scenario_num)    # initialize with 1
    
    def load_data(self) -> np.ndarray:
        """Load all data under the path. """
        if os.path.exists(self.npy_path):   # load from .npy file
            data = np.load(self.npy_path)
            data = data[:, :-1]     # the last column is the predicted label and is not needed here
            self.max_timestep = int(np.max(data[:, ::6]))
            for subpath in os.listdir(self.path):
                bv_num = int(subpath[-1])
                self.max_bv_num = max(self.max_bv_num, bv_num)
                n = 0
                for filename in os.listdir(self.path+subpath):
                    n += 1
                self.type_count[bv_num] = n
        else:   # load from original .csv files
            data = []
            for subpath in os.listdir(self.path):
                bv_num = int(subpath[-1])
                self.max_bv_num = max(self.max_bv_num, bv_num)
                n = 0
                for filename in os.listdir(self.path+subpath):
                    scenario = np.loadtxt(self.path+subpath+'/'+filename, skiprows=1, delimiter=',', dtype=float)
                    self.max_timestep = max(self.max_timestep, int(scenario[-1, 0]))
                    data.append(scenario)
                    n += 1
                self.type_count[bv_num] = n
            data = self.fill_zero(data)
        self.max_timestep += 1  # starting from 0
        return data
    
    def fill_zero(self, data: list[np.ndarray]) -> np.ndarray:
        """
        Fill 0 in the vacant part of other scenarios based on the largest dimension of all input scenarios. 

        Return an array of scenarios filled with 0 on the input scenarios. 
        Each scenario is flattened into one dimension. 
        So the shape of each scenario is (max_dim,), and the shape of the whole array is (scenario_num, max_dim).
        """
        r_max = 0
        for i in range(len(data)):
            r = data[i].shape[0]
            if r > r_max:
                r_max = r
        
        for i in range(len(data)):
            r = data[i].shape[0]
            data[i] = np.pad(data[i], ((0,r_max-r),(0,0)), 'constant', constant_values=(0,0)).flatten()
        return np.array(data)

    def sample(self, size: int) -> np.ndarray:
        """
        Randomly sample some scenarios from the Scenario Library. 

        The sampled scenarios are used to evaluate AV model and train the Reward Predictor. 

        Return an array of index. 
        """
        return np.random.randint(self.scenario_num, size=size)
    
    def labeling(self, predictor: predictor) -> None:
        """Label each scenario using the Difficulty Predictor. """
        predictor.eval()
        with torch.no_grad():
            for i in range(self.scenario_num):
                scenario = torch.tensor(self.data[i], dtype=torch.float32, device=predictor.device).unsqueeze(dim=0)
                label = predictor(scenario).item()
                self.labels[i] = label
    
    def select(self, size: int) -> np.ndarray:
        """
        Sample scenarios with their labels as probability or weight. 

        The selected scenarios is used to train the AV model. 

        Return an array of index. 
        """
        # labels = np.stack((np.arange(self.total_num), self.labels))
        # labels = labels[:, np.argsort(labels[1])]   # sort by label
        # step = math.floor(self.total_num / (size-1))
        # index = labels[0, ::step].astype(int)
        # return index
        p = self.labels / np.sum(self.labels)
        return np.random.choice(self.scenario_num, size=size, replace=False, p=p)
    
    def visualize_label_distribution(self, train_size: int, save_path='./figure/') -> None:
        """Visualize the distribution of labels using histogram. """
        plt.hist(self.labels, bins=200, density=True, histtype='barstacked', label='all label', alpha=0.8)
        select_labels = self.labels[self.select(size=train_size)]
        plt.hist(select_labels, bins=20, density=True, label='selected label', alpha=0.5)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path+'label_distribution.png')
        else:
            plt.show()
