import os
import math
import numpy as np
import torch
from utils.predictor import predictor


class scenario_lib:
    def __init__(self, path='./scenario_lib/') -> None:
        self.path = path
        self.max_bv_num = 0                     # initialize with 0
        self.type_count = []                    # count separately based on the number of bv
        self.data = self.load_data()
        self.total_num = self.data.shape[0]
        self.max_dim = self.data.shape[1]
        self.labels = np.zeros(self.total_num)  # initialize with 0
    
    def load_data(self) -> np.ndarray:
        """Load all data under the path. """
        data = []
        for subpath in os.listdir(self.path):
            self.max_bv_num = max(self.max_bv_num, int(subpath[-1]))
            n = 0
            for filename in os.listdir(self.path+subpath):
                scenario = np.loadtxt(self.path+subpath+'/'+filename, skiprows=1, delimiter=',', dtype=float)
                data.append(scenario)
                n += 1
            self.type_count.append(n)
        
        data = self.fill_zero(data)
        return np.array(data)
    
    def fill_zero(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Fill 0 in the vacant part of other scenarios based on the largest dimension of all input scenarios. 

        Return a list of scenarios filled with 0 on the input scenarios. 
        Each scenario is flattened into one dimension. 
        So the shape of each scenario is (max_dim, ). 
        """
        r_max = 0
        for i in range(len(data)):
            r = data[i].shape[0]
            if r > r_max:
                r_max = r
        
        for i in range(len(data)):
            r = data[i].shape[0]
            data[i] = np.pad(data[i], ((0,r_max-r),(0,0)), 'constant', constant_values=(0,0)).flatten()
        return data

    def sample(self, size: int) -> np.ndarray:
        """
        Randomly sample some scenarios from the Scenario Library. 

        The sampled scenarios are used to evaluate AV model and train the Reward Predictor. 

        Return an array of index. 
        """
        return np.random.randint(self.total_num, size=size)
    
    def labeling(self, predictor: predictor) -> None:
        """Label each scenario using the Difficulty Predictor. """
        for i in range(self.total_num):
            label = predictor(torch.FloatTensor(self.data[i]).unsqueeze(dim=0)).item()
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
        return np.random.choice(self.total_num, size=size, replace=False, p=p)
