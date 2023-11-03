import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from utils.scenario_lib import scenario_lib
from utils.predictor import predictor_dnn

np.random.seed(71)
name = '20230826-0159-fixed_alpha=0.15-seed=71'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')
predictor = predictor_dnn(input_dim=lib.max_dim, device=device)
predictor.to(device)
sample_index = lib.sample(size=1024)
bin_edges = np.linspace(0, 1, 21)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# fig1 = plt.figure(figsize=(6, 3))
# axes = fig1.subplots(1, 2)
fig = plt.figure(figsize=(11.5, 3.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

# before training
predictor.load_state_dict(torch.load('./model/'+name+'/round1_predictor.pth', map_location=device))
lib.labeling(predictor)
select_labels = lib.labels[lib.select(size=4096)]
sample_labels = lib.labels[lib.sample(size=4096)]
sample_labels_before = lib.labels[sample_index]
hist_all, _ = np.histogram(lib.labels, bins=bin_edges, density=True)
hist_select, _ = np.histogram(select_labels, bins=bin_edges, density=True)
ratio = hist_select / hist_all

ax0 = fig.add_subplot(gs[0, 1])
ax0.bar(bin_centers, hist_all, width=0.05, label='random sample', alpha=0.5)
ax0.bar(bin_centers, hist_select, width=0.05, label='weighted sample', alpha=0.5)
ax0_1 = ax0.twinx()
ax0_1.plot(bin_centers, ratio, color='red', label='ratio')
lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
ax0.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.15, 1.0))
ax0.set_xticks([0, 0.5, 1])
ax0.set_ylim(0, 6)
ax0_1.set_ylim(0, 4)
ax0_1.set_yticks([])
ax0.set_xlabel('Label\n(b)')
plt.title('Before Training')
# plt.savefig('./figure/label_distribution_before.pdf', bbox_inches='tight')

# after training
predictor.load_state_dict(torch.load('./model/'+name+'/round10_predictor.pth', map_location=device))
lib.labeling(predictor)
select_labels = lib.labels[lib.select(size=4096)]
sample_labels = lib.labels[lib.sample(size=4096)]
sample_labels_after = lib.labels[sample_index]
hist_all, _ = np.histogram(lib.labels, bins=bin_edges, density=True)
hist_select, _ = np.histogram(select_labels, bins=bin_edges, density=True)
ratio = hist_select / hist_all

ax1 = fig.add_subplot(gs[0, 2])
ax1.bar(bin_centers, hist_all, width=0.05, label='random sample', alpha=0.5)
ax1.bar(bin_centers, hist_select, width=0.05, label='weighted sample', alpha=0.5)
ax1_1 = ax1.twinx()
ax1_1.plot(bin_centers, ratio, color='red', label='ratio')
# lines = []
# labels = []
# for ax in fig.axes:
#     axLine, axLabel = ax.get_legend_handles_labels()
#     lines.extend(axLine)
#     labels.extend(axLabel)
# axes[1].legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 1.0))
ax1.set_ylim(0, 6)
ax1_1.set_ylim(0, 4)
ax1.set_xticks([0, 0.5, 1])
ax1.set_yticks([])
ax1_1.set_yticks([0, 1, 2, 3, 4])
ax1.set_xlabel('Label\n(c)')
plt.title('After Training')
# plt.savefig('./figure/label_distribution_after.pdf', bbox_inches='tight')

plt.subplots_adjust(wspace=0.05)
# fig1.savefig('./figure/label_distribution.pdf', bbox_inches='tight')

# scatter plot
# fig2, ax = plt.subplots()
ax = fig.add_subplot(gs[0, 0])
ax.scatter(sample_labels_before, sample_labels_after, s=5, alpha=0.5)
ax.plot([0, 1], [0, 1], '--', color='black', linewidth=1)
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('Label Before Training\n(a)')
ax.set_ylabel('Label After Training')
# plt.savefig('./figure/label_scatter.pdf', bbox_inches='tight')

# gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
plt.savefig('./figure/label.pdf', bbox_inches='tight')
