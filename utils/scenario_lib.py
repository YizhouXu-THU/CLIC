import os
import random
import numpy as np


class scenario_lib:
    def __init__(self, path='./scenario_lib/') -> None:
        self.path = path
        self.num = len(os.listdir(path))
        self.data = self.load(list(range(self.num)))
    
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
            data[i] = np.pad(data[i], ((0,r_max-r),(0,0)), 'constant', constant_values=(0,0))
        
        return data

    def sample(self, batch_size: int) -> list[int]:
        return random.sample(list(range(self.num)), batch_size)


if __name__ == '__main__':
    lib = scenario_lib()
    data_sample = lib.data[lib.sample(batch_size=3)]
