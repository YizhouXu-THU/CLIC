import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


x = np.arange(6)
y1 = [79.76, 62.3, 90.3, 0.147, 0.0050]
y2 = [75.35, 55.5, 87.4, 0.187, 0.0060]
y3 = [76.40, 58.4, 87.2, 0.177, 0.0058]
y4 = [77.53, 60.3, 87.9, 0.165, 0.0059]
y5 = [76.24, 58.0, 87.4, 0.175, 0.0058]
y6 = [74.18, 55.1, 86.0, 0.194, 0.0075]
y = np.array([y1, y2, y3, y4, y5, y6]).T
err1 = [1.67, 3.1, 1.6, 0.015, 0.0005]
err2 = [2.73, 4.6, 2.9, 0.025, 0.0008]
err3 = [1.09, 2.4, 1.5, 0.009, 0.0004]
err4 = [2.47, 3.8, 2.3, 0.023, 0.0014]
err5 = [1.88, 3.7, 2.6, 0.016, 0.0005]
err6 = [4.05, 2.8, 4.7, 0.036, 0.0014]
err = np.array([err1, err2, err3, err4, err5, err6]).T

err_attr = {'elinewidth':1, 'ecolor':'black', 'capsize':2}
bar_width = 1

fig = plt.figure(figsize=(16.5, 2.5))
axes = fig.subplots(1, 5)

axes[0].bar(x[0], y[0,0], yerr=err[0,0], error_kw=err_attr, width=bar_width, alpha=0.8, label='CLIC', 
            edgecolor='black', linewidth=2, zorder=2)
axes[0].bar(x[1], y[0,1], yerr=err[0,1], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand')
axes[0].bar(x[2], y[0,2], yerr=err[0,2], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand+fail')
axes[0].bar(x[3], y[0,3], yerr=err[0,3], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ fail')
axes[0].bar(x[4], y[0,4], yerr=err[0,4], error_kw=err_attr, width=bar_width, alpha=0.8, label='PCL')
axes[0].bar(x[5], y[0,5], yerr=err[0,5], error_kw=err_attr, width=bar_width, alpha=0.8, label='PER')
axes[0].plot([-0.7,5.7], [61.99,61.99], '--', color='black', linewidth=1, label='before training')
axes[0].set_xticks([])
axes[0].set_yticks([60, 65, 70, 75, 80])
axes[0].set_title(r'SR(%)$\uparrow$')
axes[0].set_ylim(60, 83)

axes[1].bar(x[0], y[1,0], yerr=err[1,0], error_kw=err_attr, width=bar_width, alpha=0.8, label='CLIC', 
            edgecolor='black', linewidth=2, zorder=2)
axes[1].bar(x[1], y[1,1], yerr=err[1,1], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand')
axes[1].bar(x[2], y[1,2], yerr=err[1,2], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand+fail')
axes[1].bar(x[3], y[1,3], yerr=err[1,3], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ fail')
axes[1].bar(x[4], y[1,4], yerr=err[1,4], error_kw=err_attr, width=bar_width, alpha=0.8, label='PCL')
axes[1].bar(x[5], y[1,5], yerr=err[1,5], error_kw=err_attr, width=bar_width, alpha=0.8, label='PER')
axes[1].set_xticks([])
axes[1].set_yticks([50, 55, 60, 65])
axes[1].set_title(r'FNR(%)$\uparrow$')
axes[1].set_ylim(50, 67)

axes[2].bar(x[0], y[2,0], yerr=err[2,0], error_kw=err_attr, width=bar_width, alpha=0.8, label='CLIC', 
            edgecolor='black', linewidth=2, zorder=2)
axes[2].bar(x[1], y[2,1], yerr=err[2,1], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand')
axes[2].bar(x[2], y[2,2], yerr=err[2,2], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand+fail')
axes[2].bar(x[3], y[2,3], yerr=err[2,3], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ fail')
axes[2].bar(x[4], y[2,4], yerr=err[2,4], error_kw=err_attr, width=bar_width, alpha=0.8, label='PCL')
axes[2].bar(x[5], y[2,5], yerr=err[2,5], error_kw=err_attr, width=bar_width, alpha=0.8, label='PER')
axes[2].set_xticks([])
axes[2].set_yticks([80, 85, 90])
axes[2].set_title(r'TNR(%)$\uparrow$')
axes[2].set_ylim(80, 93)

axes[3].bar(x[0], y[3,0], yerr=err[3,0], error_kw=err_attr, width=bar_width, alpha=0.8, label='CLIC', 
            edgecolor='black', linewidth=2, zorder=2)
axes[3].bar(x[1], y[3,1], yerr=err[3,1], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand')
axes[3].bar(x[2], y[3,2], yerr=err[3,2], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand+fail')
axes[3].bar(x[3], y[3,3], yerr=err[3,3], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ fail')
axes[3].bar(x[4], y[3,4], yerr=err[3,4], error_kw=err_attr, width=bar_width, alpha=0.8, label='PCL')
axes[3].bar(x[5], y[3,5], yerr=err[3,5], error_kw=err_attr, width=bar_width, alpha=0.8, label='PER')
axes[3].plot([-0.7,5.7], [0.301,0.301], '--', color='black', linewidth=1, label='before training')
axes[3].set_xticks([])
axes[3].set_yticks([0.1, 0.15, 0.2, 0.25, 0.3])
axes[3].set_title(r'CPS(s$^{-1}$)$\downarrow$')
axes[3].set_ylim(0.1, 0.32)
axes[3].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
axes[3].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

axes[4].bar(x[0], y[4,0], yerr=err[4,0], error_kw=err_attr, width=bar_width, alpha=0.8, label='CLIC', 
            edgecolor='black', linewidth=2, zorder=2)
axes[4].bar(x[1], y[4,1], yerr=err[4,1], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand')
axes[4].bar(x[2], y[4,2], yerr=err[4,2], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ rand+fail')
axes[4].bar(x[3], y[4,3], yerr=err[4,3], error_kw=err_attr, width=bar_width, alpha=0.8, label='SAC w/ fail')
axes[4].bar(x[4], y[4,4], yerr=err[4,4], error_kw=err_attr, width=bar_width, alpha=0.8, label='PCL')
axes[4].bar(x[5], y[4,5], yerr=err[4,5], error_kw=err_attr, width=bar_width, alpha=0.8, label='PER')
axes[4].plot([-0.7,5.7], [0.0112,0.0112], '--', color='black', linewidth=1, label='before training')
axes[4].set_xticks([])
axes[4].set_yticks([0.004, 0.006, 0.008, 0.01, 0.012])
axes[4].set_title(r'CPM(m$^{-1}$)$\downarrow$')
axes[4].set_ylim(0.004, 0.012)
axes[4].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
axes[4].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# axes[4].yaxis.get_offset_text().set_x(0)
# axes[4].yaxis.get_offset_text().set_y(-0.5)

lg = axes[0].legend(bbox_to_anchor=(2.8, -0.05), loc='upper center', ncol=7)
# bold_font = FontProperties(weight='bold')
# lg.texts[1].set_fontproperties(bold_font)
# txt1 = fig.text(s='We expect higher values for SR, FNR and TNR, and lower values for CPS and CPM. ', 
#                 x=0.12, y=-0.12, ha='left', va='center', fontsize=10, color='black')
# txt2 = fig.text(s='Results are from 5 random seeds. ', 
#                 x=0.12, y=-0.2, ha='left', va='center', fontsize=10, color='black')

plt.savefig('./figure/comparison.pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
