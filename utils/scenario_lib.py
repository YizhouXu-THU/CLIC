import os
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class scenario_lib:
    def __init__(self, path='./data/all/', npy_path='./data/all.npy') -> None:
        self.path = path
        self.npy_path = npy_path
        self.max_bv_num = 0                         # initialize with 0
        self.max_timestep = 0                       # initialize with 0
        self.type_count = {}                        # count separately based on the number of BV
        self.data = self.load_data()
        self.scenario_num = self.data.shape[0]
        self.max_dim = self.data.shape[1]
        self.labels = np.ones(self.scenario_num)    # initialize with 1
    
    def load_data(self) -> np.ndarray:
        """Load all data under the path. """
        if os.path.exists(self.npy_path):   # load from .npy file
            data = np.load(self.npy_path)
            data = data[:, :-1]     # the last column is the predicted labels and is not needed here
            self.max_timestep = int(np.max(data[:, ::6]))
            for subpath in os.listdir(self.path):
                bv_num = int(subpath[-1])
                self.max_bv_num = max(self.max_bv_num, bv_num)
                n = 0
                for filename in os.listdir(self.path+subpath):
                    n += 1
                self.type_count[bv_num] = n
        else:   # load from original .csv files
            data_dict = {}
            for subpath in os.listdir(self.path):
                bv_num = int(subpath[-1])
                self.max_bv_num = max(self.max_bv_num, bv_num)
                n = 0
                for filename in os.listdir(self.path+subpath):
                # for filename in tqdm(os.listdir(self.path+subpath), unit='file'):
                    scenario = np.loadtxt(self.path+subpath+'/'+filename, skiprows=1, delimiter=',', dtype=float)
                    self.max_timestep = max(self.max_timestep, int(scenario[-1, 0]))
                    if n == 0:
                        data_dict[bv_num] = [scenario]
                    else:
                        data_dict[bv_num].append(scenario)
                    n += 1
                self.type_count[bv_num] = n
            data = sorted(data_dict.items(), key=lambda x: x[0])    # sort by the number of BV
            data_list = []
            for _, value in data:
                data_list.extend(value)
            data = self.fill_zero(data_list)
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
    
    def labeling(self, predictor) -> None:
        """
        Label each scenario using the Difficulty Predictor. 
        
        The label value is a real number within [0,1]. 
        """
        predictor.eval()
        with torch.no_grad():
            scenarios = torch.tensor(self.data, dtype=torch.float32, device=predictor.device)
            labels = predictor(scenarios).cpu().numpy()
            self.labels = labels
    
    def select(self, size: int) -> np.ndarray:
        """
        Sample scenarios with their labels as probability or weight. 

        The selected scenarios is used to train the AV model. 

        Return an array of index. 
        """
        # labels = np.stack((np.arange(self.scenario_num), self.labels))
        # labels = labels[:, np.argsort(labels[1])]   # sort by label
        # step = math.floor(self.scenario_num / (size-1))
        # index = labels[0, ::step].astype(int)
        # return index
        self.labels += 1e-6
        p = self.labels / np.sum(self.labels)
        return np.random.choice(self.scenario_num, size=size, replace=False, p=p)
    
    def select_bv_num(self, bv_nums: list[int], size: int) -> np.ndarray:
        """
        Randomly select in the scenarios corresponding to the given bv_num. 

        The selected scenarios is used to train the AV model. 

        Return an array of index. 
        """
        num = 0
        for bv_num in bv_nums:
            num += self.type_count[bv_num]
        return np.random.randint(num, size=size)
    
    def visualize_label_distribution(self, num_select: int, num_sample: int, 
                                     save_path='./figure/', filename='label_distribution.png', 
                                     title='Label Distribution') -> None:
        """Visualize the distribution of all labels and the labels selected using histogram. """
        select_labels = self.labels[self.select(size=num_select)]
        sample_labels = self.labels[self.sample(size=num_sample)]
        
        # def parzen_window(x: float, X: np.ndarray, h: float) -> np.ndarray:
        #     k = 1 / (np.sqrt(2*np.pi)*h) * np.exp(-((X-x)**2)/(2*h**2))
        #     return np.average(k)
        # all_pdf_est = lambda x: np.array([parzen_window(x[i], self.labels, h=0.5) for i in range(len(x))])
        
        plt.figure()
        # x0 = np.arange(0, 1.01, 0.01)
        # all_pdf = all_pdf_est(x0)
        # plt.fill_between(x0, all_pdf, label='all label', alpha=0.2)
        plt.hist(self.labels, bins=100, density=True, label='all label', alpha=0.8)
        plt.hist(select_labels, bins=20, density=True, label='selected label', alpha=0.5)
        plt.hist(sample_labels, bins=50, density=True, label='sampled label', alpha=0.5)
        
        plt.legend(loc='upper center')
        plt.xlabel('Label')
        plt.title(title)
        if save_path is not None:
            plt.savefig(save_path + filename)
        else:
            plt.show()
