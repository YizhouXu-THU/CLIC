import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

result = np.array([[0.275, 0.408, 0.501, 0.630, 0.647, 0.706, 0.703, 0.718, 0.684, 0.737],
                   [0.178, 0.360, 0.491, 0.625, 0.643, 0.697, 0.690, 0.747, 0.682, 0.678],
                   [0.232, 0.299, 0.482, 0.570, 0.654, 0.656, 0.680, 0.704, 0.679, 0.668],
                   [0.342, 0.406, 0.389, 0.586, 0.678, 0.676, 0.694, 0.720, 0.682, 0.713],
                   [0.274, 0.324, 0.362, 0.454, 0.618, 0.554, 0.605, 0.638, 0.617, 0.645],
                   [0.241, 0.326, 0.448, 0.525, 0.493, 0.587, 0.573, 0.648, 0.607, 0.659],
                   [0.277, 0.323, 0.406, 0.439, 0.518, 0.459, 0.533, 0.575, 0.565, 0.617],
                   [0.233, 0.297, 0.420, 0.463, 0.538, 0.507, 0.512, 0.610, 0.590, 0.609],
                   [0.302, 0.334, 0.422, 0.465, 0.513, 0.486, 0.523, 0.505, 0.553, 0.613],
                   [0.295, 0.355, 0.415, 0.509, 0.547, 0.565, 0.589, 0.607, 0.518, 0.621]])

av = np.linspace(1, 10, 10, dtype=int)
predictor = np.linspace(1, 10, 10, dtype=int)
axis1, axis2 = np.meshgrid(av, predictor)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(axis1, axis2, result, cmap='turbo', edgecolor='none')
ax.set_xlabel('AV iteration')
ax.set_ylabel('predictor iteration')
ax.set_zlabel('Success Rate')
ax.view_init(elev=30, azim=135)
fig.colorbar(surf, shrink=0.8, aspect=15, pad=0.15)
plt.savefig('./figure/3D.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(result, cmap='turbo', ax=ax, cbar=True, xticklabels=av, yticklabels=predictor, annot=True, fmt='.3f')
ax.set_xlabel('AV iteration')
ax.set_ylabel('predictor iteration')
ax.set_title('Success Rate')
plt.savefig('./figure/heat_map.pdf', bbox_inches='tight')
