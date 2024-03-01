import os
import math
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch


class scenario_lib:
    def __init__(self, path='./data/all.npz') -> None:
        self.path = path
        self.max_bv_num = 0                         # initialize with 0
        self.max_timestep = 0                       # initialize with 0
        self.type_count = {}                        # count separately based on the number of BV
        self.data = self.load_data()                # shape: (scenario_num, max_dim)
        self.scenario_num = self.data.shape[0]
        self.max_dim = self.data.shape[1]
        self.min_x = np.min(self.data[:, 2::6]).item()
        self.max_x = np.max(self.data[:, 2::6]).item()
        self.min_y = np.min(self.data[:, 3::6]).item()
        self.max_y = np.max(self.data[:, 3::6]).item()
        self.max_vel = np.max(self.data[:, 4::6]).item()
        self.max_yaw = np.max(np.abs(self.data[:, 5::6])).item()
        self.labels = np.ones(self.scenario_num)    # initialize with 1
    
    def load_data(self) -> np.ndarray:
        """Load all data under the path. """
        if (os.path.exists(self.path)) and (os.path.isfile(self.path)):
            # load from NPZ file
            npz_data = np.load(self.path, allow_pickle=True)
            data = npz_data['scenario']
            self.max_timestep = int(np.max(data[:, ::6]))
            self.type_count = npz_data['type_count'].item()
            self.max_bv_num = npz_data['max_bv_num'].item()
        elif (os.path.exists(self.path)) and (os.path.isdir(self.path)):
            # load from original CSV files
            data_dict = {}
            for subpath in os.listdir(self.path):
                bv_num = int(subpath[-1])
                self.max_bv_num = max(self.max_bv_num, bv_num)
                n = 0
                # for filename in os.listdir(self.path+subpath):
                for filename in tqdm(os.listdir(self.path+subpath), unit='file'):
                    scenario = np.loadtxt(self.path+subpath+'/'+filename, skiprows=1, delimiter=',', dtype=np.float16)
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
        self.max_timestep += 1  # starting from 0, so the number of frames should + 1
        return data.astype(np.float16)
    
    def fill_zero(self, data: list[np.ndarray]) -> np.ndarray:
        """
        Fill 0 in the vacant part of other scenarios by timestep based on the largest dimension of all scenarios. 

        Return an array of scenarios filled with 0 on the input scenarios. 
        Each scenario is flattened into one dimension. 
        So the shape of each scenario is (max_dim,), and the shape of the whole array is (scenario_num, max_dim).
        """
        new_data = []
        for i in trange(len(data)):
            scenario = data[i]
            timestep = int(np.max(scenario[:, 0]))
            bv_num = int(np.max(scenario[:, 1]))
            new_scenario = np.zeros(((1 + self.max_bv_num * (self.max_timestep+1)), 6))
            new_scenario[0] = data[i][0]
            for j in range(timestep+1):
                for k in range(bv_num):
                    new_scenario[j*self.max_bv_num+k+1] = scenario[j*bv_num+k+1]
            new_data.append(new_scenario.flatten())
        return np.array(new_data)

    def sample(self, size: int) -> np.ndarray:
        """
        Randomly sample some scenarios from the Scenario Library. 

        The sampled scenarios are used to evaluate AV model and train the Reward Predictor. 

        Return an array of index. 
        """
        return np.random.randint(self.scenario_num, size=size)
    
    def labeling(self, predictor) -> np.ndarray:
        """
        Label each scenario using the Difficulty Predictor. 
        
        The label value is a real number within [0,1]. 
        """
        predictor.eval()
        with torch.no_grad():
            if predictor.__class__.__name__ == 'predictor_mlp':
                scenarios = torch.tensor(self.data, dtype=torch.float32, device=predictor.device)
                labels = predictor(scenarios)
            elif predictor.__class__.__name__ in ['predictor_rnn', 'predictor_lstm']:
                bathc_size = 128
                batch_num = math.ceil(self.scenario_num / bathc_size)
                labels = torch.zeros(0, dtype=torch.float32, device=predictor.device)
                for i in range(batch_num):
                    scenarios = self.data[i*bathc_size:(i+1)*bathc_size]
                    scenarios = torch.tensor(scenarios, dtype=torch.float32, device=predictor.device)
                    labels = torch.cat((labels, predictor(scenarios)), dim=0)
        self.labels = labels.cpu().numpy()
        return self.labels
    
    def labeling_vae(self, vae, classifier) -> np.ndarray:
        """
        Label each scenario using the VAE and Classifier. 
        
        The label value is a real number within [0,1]. 
        """
        vae.eval(), classifier.eval()
        with torch.no_grad():
            scenarios = torch.tensor(self.data, dtype=torch.float32, device=vae.device)
            _, latent, _ = vae(scenarios)
            labels = classifier(latent).cpu().numpy()
        self.labels = labels
        return labels
    
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
    
    def scenario_normalize(self, scenario: np.ndarray) -> np.ndarray:
        """Normalize the input scenario by column to facilitate scenarios reconstruction through VAE and visualization. """
        batch_size = scenario.shape[0]
        if scenario.ndim == 2:
            scenario = scenario.reshape((batch_size, -1, 6))
        
        def normalize(data: np.ndarray, min: float, max: float) -> np.ndarray:
            return (data - min) / (max - min)
        
        scenario[:, :, 0] = normalize(scenario[:, :, 0], min=0, max=self.max_timestep)
        scenario[:, :, 1] = normalize(scenario[:, :, 1], min=0, max=self.max_bv_num)
        scenario[:, :, 2] = normalize(scenario[:, :, 2], min=self.min_x, max=self.max_x)
        scenario[:, :, 3] = normalize(scenario[:, :, 3], min=self.min_y, max=self.max_y)
        scenario[:, :, 4] = normalize(scenario[:, :, 4], min=0.0, max=self.max_vel)
        scenario[:, :, 5] = normalize(scenario[:, :, 5], min=-self.max_yaw, max=self.max_yaw)

        return scenario.reshape((batch_size, -1))
    
    def scenario_denormalize(self, scenario: np.ndarray) -> np.ndarray:
        """Denormalize the input scenario by column. """
        batch_size = scenario.shape[0]
        if scenario.ndim == 2:
            scenario = scenario.reshape((batch_size, -1, 6))
        
        def denormalize(data: np.ndarray, min: float, max: float) -> np.ndarray:
            return data * (max - min) + min
        
        scenario[:, :, 0] = denormalize(scenario[:, :, 0], min=0, max=self.max_timestep)
        scenario[:, :, 1] = denormalize(scenario[:, :, 1], min=0, max=self.max_bv_num)
        scenario[:, :, 2] = denormalize(scenario[:, :, 2], min=self.min_x, max=self.max_x)
        scenario[:, :, 3] = denormalize(scenario[:, :, 3], min=self.min_y, max=self.max_y)
        scenario[:, :, 4] = denormalize(scenario[:, :, 4], min=0.0, max=self.max_vel)
        scenario[:, :, 5] = denormalize(scenario[:, :, 5], min=-self.max_yaw, max=self.max_yaw)

        return scenario.reshape((batch_size, -1))
