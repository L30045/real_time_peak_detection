# %% Import third party library for visualization
import numpy as np
import matplotlib.pyplot as plt
import dependencies as dep
import time

# %% load file
filepath = 'C:\\Users\\Yuan\\OneDrive\\Desktop\\Algorithm questionnaire\\2 algorithm Data set\\Q1\\'
filename = 'Q1 Dataset 1.txt'
savename = 'O1 Answer.txt'
times = []
raw_data = []
with open(filepath+filename,'r') as f:
    for line in f:
        word = line.split('\t')
        if len(word) == 2 and word[0][0]!='T':
            times.append(float(word[0]))
            raw_data.append(float(word[1].split()[0]))

# %% visualize data for intuition
fig, ax = plt.subplots(2,1)
ax[0].plot(times, raw_data)
plt.grid()
ax[1].plot(times[5800:6200],raw_data[5800:6200])
plt.grid()
plt.xlabel('Times')

# %% extract basic features
# check if data missing
t_diff = [x-y for x, y in zip(times[2:],times[1:-1])]
if len(set(t_diff))!=1:
    print('Data missing.')
else:
    # srate
    srate = 1/t_diff[0]
    print(f'Data correct. Sampling rate = {srate:.1f} Hz')

# %% simulated stream
len_bus = 1 # sec
len_bus_smpl = int(round(len_bus*srate))
# bus_data, bus_time = dep.sim_stream(raw_data[6200:6700],times[6200:6700],len_bus_smpl)
bus_data, bus_time = dep.sim_stream(raw_data,times,len_bus_smpl)

#%% test detrend
len_win = 500
z_thres = 2.5
nb_peakpoint = 10
thres_method = 'z-score'
pd = dep.peak_detection(threshold=z_thres,len_win=len_win,peak_decay_threshold=nb_peakpoint,thres_method=thres_method)
mv_mu = []
mv_std = []
output_value = []
output_time = []
pred_val = []
time_cost = []

for i, (tmp_data, tmp_time) in enumerate(zip(bus_data,bus_time)):
    start_time = time.time()
    loc_val, loc_t = pd.peak_detect(tmp_data[0],tmp_time[0])
    time_cost.append(time.time()-start_time)
    output_value.append(loc_val)
    output_time.append(loc_t)
    mv_mu.append(pd.filtered_val)
    mv_std.append(pd.mv_std)
    pred_val.append(pd.pred_val)

#%% report output
with open(filepath+savename,'w') as f:
    f.write('position\theight\n')
    for peak_info in zip(output_value,output_time):
        p_val = peak_info[0]
        p_t = peak_info[1]
        if p_val!=0:
            f.write(str(p_t)+'\t'+str(p_val)+'\n')

#%% visualize regression
plt.figure(figsize=[20,10])
plt.plot(times,mv_mu,label='mv_mu')
plt.plot(times,raw_data,label='raw')
plt.plot(times,pred_val,label='pred')
plt.plot(output_value,label='peak')
plt.legend(fontsize=20)

#%% visualize regression
plt.figure(figsize=[20,10])
plt.plot(times[6200:6700],mv_mu[6200:6700],label='mv_mu')
plt.plot(times[6200:6700],raw_data[6200:6700],label='raw')
plt.plot(times[6200:6700],pred_val[6200:6700],label='pred')
idx_peak = np.nonzero(output_time[6200:6700])
plt.plot(output_time[6200+idx_peak[0][0]],output_value[6200+idx_peak[0][0]]+pred_val[6200+idx_peak[0][0]],'*',label='peak',color='red',markersize=20)
plt.legend(fontsize=20)
plt.xlim([6200,6700])
plt.ylim([39,44])

#%% visualize regression
plt.figure(figsize=[20,10])
plt.plot(times[4500:5500],mv_mu[4500:5500],label='mv_mu')
plt.plot(times[4500:5500],raw_data[4500:5500],label='raw')
plt.plot(times[4500:5500],pred_val[4500:5500],label='pred')
idx_peak = np.nonzero(output_time[4500:5500])
plt.plot(output_time[4500+idx_peak[0][0]],output_value[4500+idx_peak[0][0]]+pred_val[4500+idx_peak[0][0]],'*',label='peak',color='red',markersize=20)
plt.plot(output_time[4500+idx_peak[0][1]],output_value[4500+idx_peak[0][1]]+pred_val[4500+idx_peak[0][1]],'*',label='peak',color='red',markersize=20)
# plt_output = [x+y for x, y in zip(output_value[4500:5500],pred_val[4500:5500])]
# plt.scatter(output_time[4500:5500],plt_output,label='peak',color='red',marker='*')
plt.legend(fontsize=20)
plt.xlim([4500,5500])

#%% visualize regression
plt.figure(figsize=[20,10])
plt.plot(times[12800:13200],mv_mu[12800:13200],label='mv_mu')
plt.plot(times[12800:13200],raw_data[12800:13200],label='raw')
plt.plot(times[12800:13200],pred_val[12800:13200],label='pred')
idx_peak = np.nonzero(output_time[12800:13200])
plt.plot(output_time[12800+idx_peak[0][0]],output_value[12800+idx_peak[0][0]]+pred_val[12800+idx_peak[0][0]],'*',label='peak',color='red',markersize=20)
plt.plot(output_time[12800+idx_peak[0][1]],output_value[12800+idx_peak[0][1]]+pred_val[12800+idx_peak[0][1]],'*',label='peak',color='red',markersize=20)
# # plt_output = [x+y for x, y in zip(output_value[12800:13200],mv_mu[12800:13200])]
# plt.scatter(output_time[12800:13200],plt_output,label='peak',color='red',marker='*')
plt.legend(fontsize=20)
plt.xlim([12800,13200])

#%% visualize regression
plt.figure(figsize=[20,10])
plt.plot(times[5900:6100],mv_mu[5900:6100],label='mv_mu')
plt.plot(times[5900:6100],raw_data[5900:6100],label='raw')
plt.plot(times[5900:6100],pred_val[5900:6100],label='pred')
idx_peak = np.nonzero(output_time[5900:6100])
plt.plot(output_time[5900+idx_peak[0][0]],output_value[5900+idx_peak[0][0]]+pred_val[5900+idx_peak[0][0]],'*',label='peak',color='red',markersize=20)
# plt_output = [x+y for x, y in zip(output_value[5900:6100],mv_mu[5900:6100])]
# plt.scatter(output_time[5900:6100],plt_output,label='peak',color='red',marker='*')
plt.legend(fontsize=20)
plt.xlim([5900,6100])
plt.ylim([39.5,42.5])

# %% test moving average
len_win = 500
pd = dep.peak_detection_zscore_naive(threshold=3,len_win=len_win)
mv_mu = []
mv_std = []
output_value = []
output_time = []

mu = 0
decay_rate = 0.1

for tmp_data, tmp_time in zip(bus_data,bus_time):
    loc_val, loc_t = pd.peak_detect(tmp_data[0],tmp_time[0])
    # if loc_val!=0:
    output_value.append(loc_val)
    output_time.append(loc_t)
    mv_mu.append(pd.mv_avg)
    mv_std.append(pd.mv_std)

#%%
plt.figure(figsize=[20,10])
# plt.plot(times,mv_mu)
# plt.plot(times,raw_data)
# plt.plot(output_value)
plt.plot(mv_mu-np.mean(raw_data[6200:6700]))
plt.plot(raw_data[6200:6700]-np.mean(raw_data[6200:6700]))
plt.plot([x-y for x,y in zip(raw_data[6200:6700],mv_mu)])

