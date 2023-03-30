import os
import math
import random
import numpy as np
import torch
from utils.reward_predictor import reward_predictor


class scenario_lib:
    def __init__(self, path='./scenario_lib/') -> None:
        self.path = path
        self.num = len(os.listdir(path))
        self.data = self.load(list(range(self.num)))
        self.max_dim = self.data.shape[1]
        self.labels = [0.0] * self.num  # all labels are set to 0 during initialization
    
    def load(self, index: list[int]) -> np.ndarray:
        data = []
        for i in index:
            scenario = np.loadtxt(self.path+str(i)+'.csv', delimiter=',', skiprows=1)
            data.append(scenario)
        
        data = self.fill_zero(data)
        data = np.array(data)
        return data
    
    def fill_zero(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Fill 0 in the vacant part of other scenarios based on the largest dimension of all input scenarios. 

        Return a list of scenarios filled with 0 on the input scenarios. 
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

    def sample(self, size: int) -> list[int]:
        """Return a list of index. """
        return random.sample(list(range(self.num)), size)
    
    def labeling(self, predictor: reward_predictor) -> None:
        labels = []
        for i in range(self.num):
            label = predictor(torch.FloatTensor(self.data[i]).unsqueeze(dim=0)).item()
            labels.append(label)
        self.labels = labels
    
    def select(self, size: int) -> list[int]:
        """
        Sort all scenario by label and sample evenly according to the order. 

        Return a list of index. 
        """
        # labels = np.array(self.labels)
        # index = []
        # for i in range(size):
        #     indices = np.argwhere((labels >= i/size) & (labels <= (i+1)/size))
        #     index.append(indices[random.randrange(indices.size)][0])
        # return index
        
        labels = np.stack((np.arange(self.num), np.array(self.labels)))
        labels = labels.T(labels.T[:,1].argsort()).T    # sort by label
        step = math.floor(self.num / size)
        index = labels[0, ::step].tolist()
        return index
