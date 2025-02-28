import yaml
from dataIO.data import IonoDataManager
from dataIO.dataPostprocess import get_location
import matplotlib.pyplot as plt
import numpy as np
import os
cfgs = yaml.load(open('D:/IASGAN/example_config.yaml', 'r'), Loader=yaml.BaseLoader)
dataManager = IonoDataManager(cfgs)

i = 0
test_data, human_res, art_res = dataManager.get_test_batch(i)

x_start, x_spacing = 1, 0.05
y_start, y_spacing = 80, 2.5



r_1 = get_location(human_res[0, :, :, :], 0)
g_1 = get_location(human_res[0, :, :, :], 1)
b_1 = get_location(human_res[0, :, :, :], 2)

r_2 = get_location(art_res[0, :, :, :], 0)
g_2 = get_location(art_res[0, :, :, :], 1)
b_2 = get_location(art_res[0, :, :, :], 2)

a = np.transpose(r_2)
b = np.transpose(g_2)
c = np.transpose(b_2)
image = human_res[0, :, :, :]


output_folder = f""
os.makedirs(output_folder, exist_ok=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image, extent=[
    x_start, x_start + image.shape[1] * x_spacing,
    y_start, y_start + image.shape[0] * y_spacing
    ], origin='lower', cmap='viridis')

ax.set_xlim([x_start, x_start + 250 * x_spacing])
ax.set_ylim([y_start, y_start + 250 * y_spacing])

ratio = 1
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
plt.xlabel('Critical Frequency(MHz)', fontsize=10)
plt.ylabel('Virtual Height(km)', fontsize=10)
plt.savefig(f"{output_folder}/T_{i}.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image, extent=[
        x_start, x_start + image.shape[1] * x_spacing,
        y_start, y_start + image.shape[0] * y_spacing
    ], origin='lower', cmap='viridis')
ax.scatter(a[:, 1] * x_spacing + x_start, a[:, 0] * y_spacing + y_start, c='yellow', marker='*', s=10)

ax.set_xlim([x_start + 0 * x_spacing, x_start + 100 * x_spacing])
ax.set_ylim([y_start + 0 * y_spacing, y_start + 35 * y_spacing])

ratio = 1.0
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
plt.xlabel('Critical Frequency(MHz)', fontsize=10)
plt.ylabel('Virtual Height(km)', fontsize=10)
plt.savefig(f"{output_folder}/ARTIST_{i}_E.png", bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image, extent=[
        x_start, x_start + image.shape[1] * x_spacing,
        y_start, y_start + image.shape[0] * y_spacing
    ], origin='lower', cmap='viridis')
ax.scatter(b[:, 1] * x_spacing + x_start, b[:, 0] * y_spacing + y_start, c='yellow', marker='*', s=10)

ax.axvline(x=np.max(g_2[1, :]) * x_spacing + x_start, color='white', linestyle='--', linewidth=1.5)
ax.axvline(x=np.max(g_1[1, :]) * x_spacing + x_start, color='white', linestyle='-', linewidth=1.5)

ax.axhline(y=np.min(g_2[0, :]) * y_spacing + y_start, color='black', linestyle='--', linewidth=1.5)
ax.axhline(y=np.min(g_1[0, :]) * y_spacing + y_start, color='black', linestyle='-', linewidth=1.5)
ax.set_xlim([x_start + 30 * x_spacing, x_start + 100 * x_spacing])
ax.set_ylim([y_start + 30 * y_spacing, y_start + 120 * y_spacing])

ratio = 1.0
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
plt.xlabel('Critical Frequency(MHz)', fontsize=10)
plt.ylabel('Virtual Height(km)', fontsize=10)
plt.savefig(f"{output_folder}/ARTIST_{i}_F1.png", bbox_inches='tight')
plt.close(fig)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image, extent=[
        x_start, x_start + image.shape[1] * x_spacing,
        y_start, y_start + image.shape[0] * y_spacing
    ], origin='lower', cmap='viridis')
ax.scatter(c[:, 1] * x_spacing + x_start, c[:, 0] * y_spacing + y_start, c='yellow', marker='*', s=10)

ax.axvline(x=np.max(b_2[1, :]) * x_spacing + x_start, color='white', linestyle='--', linewidth=1.5)
ax.axvline(x=np.max(b_1[1, :]) * x_spacing + x_start, color='white', linestyle='-', linewidth=1.5)

ax.axhline(y=np.min(b_2[0, :]) * y_spacing + y_start, color='black', linestyle='--', linewidth=1.5)
ax.axhline(y=np.min(b_1[0, :]) * y_spacing + y_start, color='black', linestyle='-', linewidth=1.5)


ax.set_xlim([x_start + 0 * x_spacing, x_start + 100 * x_spacing])
ax.set_ylim([y_start + 50 * y_spacing, y_start + 200 * y_spacing])

ratio = 1.0
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
plt.xlabel('Critical Frequency(MHz)', fontsize=10)
plt.ylabel('Virtual Height(km)', fontsize=10)
plt.savefig(f"{output_folder}/ARTIST_{i}_F2.png", bbox_inches='tight')
plt.close(fig)