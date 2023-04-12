import os
import csv
import math
import random
import numpy as np
import torch
from utils.reward_predictor import reward_predictor


class scenario_lib:
    def __init__(self, path='./scenario_lib/') -> None:
        self.path = path
        self.max_bv_num = 0             # initialize with 0
        self.data, self.av_speed = self.load_data()
        self.num = self.data.shape[0]
        self.max_dim = self.data.shape[1]
        self.labels = [0.0] * self.num  # initialize with 0
    
    def load_data(self) -> np.ndarray:
        """Load all data under the path. """
        # TODO: classify by BV quantity
        data = []
        av_speed = []
        for filename in os.listdir(self.path):
            with open(self.path+filename, 'r') as f:
                reader = csv.reader(f)
                scenario = []
                for i, row in enumerate(reader):
                    if i == 1:
                        av_info = np.array(row, dtype=np.float64)
                    elif i > 1:
                        scenario.append(np.array(row, dtype=np.float64))
                av_speed.append(av_info[-1])
                av_info = np.expand_dims(np.array(av_info[0:-1]), axis=0)
                scenario = np.concatenate((av_info, scenario), axis=0)
                data.append(scenario)

                max_bv_num = np.max(scenario[:, 1])
                if max_bv_num > self.max_bv_num:
                    self.max_bv_num = int(max_bv_num)
        
        data = self.fill_zero(data)
        return np.array(data), np.array(av_speed)
    
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
        """
        Randomly sample some scenarios from the scenario library. 

        The sampled scenarios are used to evaluate AV model and train the Reward Predictor. 

        Return a list of index. 
        """
        return random.sample(list(range(self.num)), size)
    
    def labeling(self, predictor: reward_predictor) -> None:
        """Label each scenario using the Reward Predictor. """
        labels = []
        for i in range(self.num):
            label = predictor(torch.FloatTensor(self.data[i]).unsqueeze(dim=0)).item()
            labels.append(label)
        self.labels = labels
    
    def select(self, size: int) -> list[int]:
        """
        Sort all scenarios by label and sample evenly according to the order. 

        The selected scenarios is used to train the AV model. 

        Return a list of index. 
        """
        # labels = np.array(self.labels)
        # index = []
        # for i in range(size):
        #     indices = np.argwhere((labels >= i/size) & (labels <= (i+1)/size))
        #     index.append(indices[random.randrange(indices.size)][0])
        # return index
        
        labels = np.stack((np.arange(self.num), np.array(self.labels)))
        labels = labels[:, np.argsort(labels[1])]   # sort by label
        step = math.floor(self.num / (size-1))
        index = labels[0, ::step].astype(int).tolist()
        return index
