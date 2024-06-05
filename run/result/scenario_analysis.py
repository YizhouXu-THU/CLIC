import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

from tqdm import trange
from math import atan2, sqrt, pi, atan2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from utils.scenario_lib import scenario_lib
from utils.function import set_random_seed

random_seed = 42    # 14, 42, 51, 71, 92
set_random_seed(random_seed)
lib = scenario_lib(path='./data/all.npz')
highd_path = './data/highD/data/'
delta_t = 0.04

# scenario library
bv_bv_dis_lib, bv_av_dis_lib, bv_av_pos_x_lib, bv_av_pos_y_lib, bv_yaw_lib, bv_vel_lib, bv_acc_lib = [], [], [], [], [], [], []
for i in trange(lib.scenario_num):
    scenario = lib.data[i].reshape((-1, 6))
    total_timestep = int(np.max(scenario[:, 0]))
    bv_num = int(np.max(scenario[:, 1]))
    for j in range(bv_num):
        bv_av_dis_lib.append(sqrt((scenario[0, 2] - scenario[1 + j, 2]) ** 2 + 
                                  (scenario[0, 3] - scenario[1 + j, 3]) ** 2))
        bv_av_pos_x_lib.append(scenario[1 + j, 2] - scenario[0, 2])
        bv_av_pos_y_lib.append(scenario[1 + j, 3] - scenario[0, 3])
        for t in range(total_timestep):
            bv_yaw_lib.append(scenario[1 + t * bv_num + j, 5] * 180 / pi)
            bv_vel_lib.append(scenario[1 + t * bv_num + j, 4])
            if t < total_timestep - 1:
                bv_acc_lib.append((scenario[1 + (t + 1) * bv_num + j, 4] - scenario[1 + t * bv_num + j, 4]) / delta_t)
            for k in range(bv_num):
                if k != j:
                    bv_bv_dis_lib.append(sqrt((scenario[1 + t * bv_num + j, 2] - scenario[1 + t * bv_num + k, 2]) ** 2 + 
                                              (scenario[1 + t * bv_num + j, 3] - scenario[1 + t * bv_num + k, 3]) ** 2))

# highD dataset
bv_bv_dis_highd, bv_yaw_highd, bv_vel_highd, bv_acc_highd = [], [], [], []
for file in os.listdir(highd_path):
    if file.endswith('tracks.csv'):
        print(file)
        data = np.loadtxt(highd_path+file, delimiter=',', skiprows=1)
        max_timestep = int(np.max(data[:, 0]))
        veh_num = int(np.max(data[:, 1]))
        for i in range(data.shape[0]):
            if data[i, 6] > 0:
                yaw = atan2(data[i, 7], data[i, 6]) * 180 / pi
            elif data[i, 6] < 0:
                yaw = -atan2(data[i, 7], -data[i, 6]) * 180 / pi
            bv_yaw_highd.append(yaw)
            bv_vel_highd.append(sqrt(data[i, 6] ** 2 + data[i, 7] ** 2))
            bv_acc_highd.append(sqrt(data[i, 8] ** 2 + data[i, 9] ** 2))
        for t in trange(max_timestep):
            scene = data[data[:, 0] == t+1]
            for i in range(scene.shape[0]):
                for j in range(i+1, scene.shape[0]):
                    bv_bv_dis_highd.append(sqrt((scene[i, 2] - scene[j, 2]) ** 2 + (scene[i, 3] - scene[j, 3]) ** 2))


plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_bv_dis_lib, dtype=np.float16), bins=50, alpha=0.8, density=True, label='scenario library')
plt.hist(np.array(bv_bv_dis_highd, dtype=np.float16), bins=50, alpha=0.8, density=True, label='highD dataset')
plt.legend()
plt.title('BV-BV distance distribution')
plt.xlabel('distance(m)')
plt.ylabel('frequency')
plt.savefig('./figure/bv_bv_dis.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_av_dis_lib, dtype=np.float16), bins=25, alpha=0.8, density=True, label='scenario library')
plt.hist(np.array(bv_bv_dis_highd, dtype=np.float16), bins=25, alpha=0.8, density=True, label='highD dataset')
plt.legend()
plt.title('BV-AV distance distribution')
plt.xlabel('distance(m)')
plt.ylabel('frequency')
plt.xlim(-8, 175)
plt.savefig('./figure/bv_av_dis.pdf', bbox_inches='tight')

index = random.sample(range(len(bv_av_pos_x_lib)), 2000)
bv_av_pos_x_lib = [bv_av_pos_x_lib[i] for i in index]
bv_av_pos_y_lib = [bv_av_pos_y_lib[i] for i in index]
plt.figure(figsize=(9, 1.4))
plt.scatter(np.array(bv_av_pos_x_lib, dtype=np.float16), np.array(bv_av_pos_y_lib, dtype=np.float16), 
            s=2, alpha=0.5, label='BV')
plt.scatter(0, 0, s=20, c='r', label='AV')
plt.grid()
plt.legend()
plt.axis('equal')
plt.title('BV-AV relative position')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.savefig('./figure/bv_av_pos.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_yaw_lib, dtype=np.float16), bins=50, alpha=0.8, density=True, label='scenario library')
plt.hist(np.array(bv_yaw_highd, dtype=np.float16), bins=50, alpha=0.8, density=True, label='highD dataset')
plt.legend()
plt.title('BV yaw distribution')
plt.xlabel(r'yaw($^\circ$)')
plt.ylabel('frequency')
plt.savefig('./figure/bv_yaw.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_vel_lib, dtype=np.float16), bins=50, alpha=0.8, density=True, label='scenario library')
plt.hist(np.array(bv_vel_highd, dtype=np.float16), bins=50, alpha=0.8, density=True, label='highD dataset')
plt.legend()
plt.title('BV velocity distribution')
plt.xlabel('velocity(m/s)')
plt.ylabel('frequency')
plt.savefig('./figure/bv_vel.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_acc_lib, dtype=np.float16), bins=50, alpha=0.8, density=True, label='scenario library')
plt.hist(np.array(bv_acc_highd, dtype=np.float16), bins=50, alpha=0.8, density=True, label='highD dataset')
plt.legend()
plt.title('BV acceration distribution')
plt.xlabel(r'acceleration(m/s$^2$)')
plt.ylabel('frequency')
plt.xlim(-9, 9)
plt.xticks(np.arange(-8, 10, 4))
plt.savefig('./figure/bv_acc.pdf', bbox_inches='tight')
