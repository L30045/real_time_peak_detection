#%%
import numpy as np
import matplotlib.pyplot as plt
# Make numpy print 4 significant digits for prettiness
np.set_printoptions(precision=4, suppress=True)
np.random.seed(5) # To get predictable random numbers

#%%
n_points = 40
x_vals = np.arange(n_points)
y_vals = np.random.normal(size=n_points)
plt.bar(x_vals, y_vals)
# %%
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

#%%
FWHM = 4
sigma = fwhm2sigma(FWHM)
x_position = 13 # 14th point
kernel_at_pos = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
kernel_at_pos = kernel_at_pos / sum(kernel_at_pos)
plt.bar(x_vals, kernel_at_pos)
# %%
smoothed_vals = np.zeros(y_vals.shape)
for x_position in x_vals:
     kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
     kernel = kernel / sum(kernel)
     smoothed_vals[x_position] = sum(y_vals * kernel)
plt.bar(x_vals, smoothed_vals)
# %%
x = np.arange(1,51).reshape(-1,1)
H = np.hstack([np.ones((50,1)),x])
y = np.arange(5,55).reshape(-1,1)+np.random.random((50,1))*10
# %%
theta = np.dot(np.dot(np.linalg.inv(np.dot(H.T,H)),H.T),y)
# %%
plt.figure
plt.plot(x,y)
plt.plot(x,np.dot(H,theta))
# %%
