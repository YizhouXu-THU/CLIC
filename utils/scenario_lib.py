import os
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
    
    def load(self, index: list[int]) -> np.ndarray:
        data = []
        for i in index:
            scenario = np.loadtxt(self.path+str(i)+'.csv', delimiter=',', skiprows=1)
            data.append(scenario)
        
        data = self.fill_zero(data)
        data = np.array(data)
        return data
    
    def fill_zero(self, data: list[np.ndarray]) -> list[np.ndarray]:
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
        """return a list of index. """
        return random.sample(list(range(self.num)), size)
    
    def labeling(self, predictor: reward_predictor) -> None:
        labels = []
        for i in range(self.num):
            label = predictor(torch.FloatTensor(self.data[i]).unsqueeze(dim=0)).item()
            labels.append(label)
        self.labels = labels
    
    def select(self, size: int) -> list[int]:
        """
        Divide the [0,1] interval into 'size' equal parts, 
        and randomly sample one scenario from each part. 

        return a list of index. 
        """
        labels = np.array(self.labels)
        index = []
        for i in range(size):
            indices = np.argwhere((labels >= i/size) & (labels <= (i+1)/size))
            index.append(indices[random.randrange(indices.size)][0])
            # TODO: consider situations where there are no scenarios within a certain interval
        return index
