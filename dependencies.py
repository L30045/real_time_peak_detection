# %% simulated stream
def sim_stream(raw_data, times=[], len_bus_smpl=10):
    """
    This function create simulated stream with desired data bus length.
    Parameters:
      raw_data: the offline data to be streamed. (assume there is only 1 channel)
      times: time stream. Default: 0 to length of raw_data
      len_bus: data bus length. unit: sample
    Return:
      bus_time: simulated streams time.
      bus_data: simulated streams data.
    TODO:
      1) handle missing time points
    """
    # load data
    if len(times)==0:
        times = [x for x in range(len(raw_data))]
    bus_idx_start = [x for x in range(0, len(raw_data), len_bus_smpl)]
    bus_idx_end = bus_idx_start[1:]
    bus_idx_end.append(len(raw_data))
    bus_time = [times[x:y] for x,y in zip(bus_idx_start,bus_idx_end)]
    bus_data = [raw_data[x:y] for x,y in zip(bus_idx_start,bus_idx_end)]
    return bus_data, bus_time

# %% Peak detection class
class peak_detection:
    """
    peak detection class
    attribute:
        threshold: amplitude-based threshold Default: 3 unit
        len_win: data length in buffer
        peak_decay_threshold: # of nearby points for peak definition
        thres_method: threshold method. amplitude or z-score. Default: z-score
    method:
        peak_detect: detect peak, report peak height and time
    """
    def __init__(self,threshold=2.5,len_win=500,peak_decay_threshold=10,thres_method='z-score'):
        self.threshold = threshold
        self.len_win = len_win
        # peak report
        self.peak_decay_threshold=peak_decay_threshold
        self.thres_method = thres_method.lower()
        self.peak_start = []
        self.peak_start_time = []
        self.local_peak = []
        self.local_peak_height = []
        self.local_peak_time = []
        # keep track of std
        self.mv_std = -1
        # buffer for moving window
        self.buffer = []
        # detrend parameters
        self.detrend_parameters = [-1,-1,-1,-1,-1]
        self.det = []
        self.detrend_offset = []
        self.detrend_slope = []
        self.detrend_weight = 1
        self.filtered_val = []
        self.pred_val = []
        # self.count = 0

    # peak detection
    def peak_detect(self, input_data, input_time):
        output_value = 0
        output_time = 0
        # initialize buffer
        if len(self.buffer)<self.len_win:
            self.buffer.append(input_data)
            self.filtered_val = input_data
            self.pred_val = input_data
            return output_value, output_time
        # buffer filled
        # =============
        # detrend
        self.detrend()
        predict_input = self.detrend_offset+self.detrend_slope*(self.len_win+1)
        self.pred_val = predict_input
        # compare amplitude
        # detect positive peak only
        val_diff = input_data-predict_input
        # choose threshold method
        if self.thres_method=='amp':
            peak_thres = self.threshold
        else:
            peak_thres = self.threshold*self.mv_std
        if val_diff > peak_thres:
            # self.count = 0
            output_value, output_time = self.find_peak(val_diff,input_data,input_time)
            # update buffer
            self.filtered_val = input_data*(1-self.detrend_weight)+predict_input*self.detrend_weight
            self.buffer.pop(0)
            self.buffer.append(self.filtered_val)
        else:
            # reset peak
            self.peak_start = []
            self.filtered_val = input_data
            # TODO: will report complex error
            # update std only when all data in buffer are raw data
            # self.count += 1
            # self.detrend_std_update()
            #=========================
            # update std
            self.detrend_std()
            # update buffer
            self.buffer.append(self.filtered_val)
            self.buffer.pop(0)

        return output_value, output_time
    
    # detrend using linear regression
    def detrend(self):
        # theta = (H.T * H)^-1 * H.T * y
        if self.detrend_parameters[0]==-1:
            # initiate regressor
            x_pos = [i+1 for i in range(self.len_win)]
            self.detrend_parameters[0] = self.len_win # h11
            self.detrend_parameters[1] = sum(x_pos) # h12, h21
            self.detrend_parameters[2] = sum(self.buffer) # sum_y
            self.detrend_parameters[3] = sum([y*(i+1) for i, y in enumerate(self.buffer)]) # sum_xy
            self.detrend_parameters[4] = sum([x**2 for x in x_pos]) # h_22
            self.det = self.detrend_parameters[0]*self.detrend_parameters[4]-self.detrend_parameters[1]**2
            self.detrend_offset = (self.detrend_parameters[2]*self.detrend_parameters[4]-self.detrend_parameters[1]*self.detrend_parameters[3])/self.det
            self.detrend_slope = (self.detrend_parameters[0]*self.detrend_parameters[3]-self.detrend_parameters[1]*self.detrend_parameters[2])/self.det
            self.detrend_std()
        else:
            # YOU ONLY NEED TO UPDATE Y!! YEAH!!
            self.detrend_parameters[2] = sum(self.buffer)
            self.detrend_parameters[3] = sum([y*(i+1) for i, y in enumerate(self.buffer)])
            self.detrend_offset = (self.detrend_parameters[2]*self.detrend_parameters[4]-self.detrend_parameters[1]*self.detrend_parameters[3])/self.det
            self.detrend_slope = (self.detrend_parameters[0]*self.detrend_parameters[3]-self.detrend_parameters[1]*self.detrend_parameters[2])/self.det

    # peak finding
    def find_peak(self,val_diff,input_data,input_time):
        output_value = 0
        output_time = 0
        # record peak event
        if not self.peak_start:
            self.peak_start = input_data
            self.peak_start_time = input_time
            self.local_peak = input_data
            self.local_peak_height = val_diff
            self.local_peak_time = input_time
        else:
            # if abs(input_time-6000) <= self.peak_decay_threshold:
            #     print(input_time,input_data,self.local_peak_height, self.local_peak,self.local_peak_time)
            # find peak
            if input_data >= self.local_peak or \
                self.local_peak_time - self.peak_start < self.peak_decay_threshold:
                # if abs(input_time-6000) <= self.peak_decay_threshold:
                #     print('enter climb')
                # track peak climbing
                self.local_peak = input_data
                self.local_peak_height = val_diff
                self.local_peak_time = input_time
            elif input_time-self.local_peak_time > self.peak_decay_threshold and\
                self.local_peak > self.peak_start and\
                self.local_peak-input_data > self.threshold*self.mv_std:
                # if abs(input_time-6000) <= self.peak_decay_threshold:
                #     print('enter decline',input_time,input_data,self.local_peak_height, self.local_peak,self.local_peak_time)
                # track peak declining
                # report criteria:
                # 1) if decline after threshold length
                # 2) local peak higher than peak start
                # 3) local peak amplitude larger than amplitude threshold comparing to nearby points
                # 4) emperical amplitude threshold (self.peak > 2) (REMOVED)
                output_value = self.local_peak_height
                output_time = self.local_peak_time
                # reset peak
                self.peak_start = []
            # reset peak start during downhill
            elif self.peak_start >= input_data:
                # if abs(input_time-6000) <= self.peak_decay_threshold:
                #     print('enter reset')
                self.peak_start = input_data
                self.peak_start_time = input_time
                self.local_peak = input_data
                self.local_peak_height = val_diff
                self.local_peak_time = input_time
        return output_value, output_time

    # useful function 
    def detrend_std(self):
        self.mv_std = (sum([(buffer_data-(self.detrend_offset+self.detrend_slope*(i+1)))**2 for i,buffer_data in enumerate(self.buffer)])/ \
            (self.len_win-1))**(0.5)
    
    # BUG: Report comlex number. Chi-2022/11/07
    def detrend_std_update(self):
        if self.count > self.len_win:
            tmp_var = self.mv_std**2
            xn_pred = (self.filtered_val-self.pred_val)**2
            x1_pred = (self.buffer[0]-(self.detrend_offset+self.detrend_slope))**2
            self.mv_std = (tmp_var + (xn_pred-x1_pred)/(self.len_win-1))**(0.5)

class peak_detection_zscore_naive:
    """
    using moving window to calculate mean and std and threshold peak by zscore
    """
    def __init__(self,threshold=3,len_win=500,peak_decay_threshold=1):
        self.threshold = threshold
        self.len_win = len_win
        # peak report
        self.peak_decay_threshold=peak_decay_threshold
        self.peak_start = []
        self.peak_end = []
        self.local_peak = []
        self.local_peak_time = []
        # keep track of mean and std
        self.mv_avg = -1
        self.mv_std = -1
        # buffer for moving window
        self.buffer = []
        self.buffer_time = []

    # peak detection
    def peak_detect(self, input_data, input_time):
        output_value = 0
        output_time = 0
        # initialize buffer
        if not self.buffer:
            self.buffer.append(input_data)
            self.buffer_time.append(input_time)
            self.mv_avg = input_data
            self.mv_std = 0
            return output_value, output_time
        # calculate mean and std
        if len(self.buffer)<self.len_win:
            self.update_buffer_mean_std(input_data)
            self.buffer.append(input_data)
            self.buffer_time.append(input_time)
        else:
            # compare amplitude
            if abs(input_data-self.mv_avg) > self.threshold*self.mv_std:
                output_value = input_data
                output_time = input_time
                # update buffer
                filtered_input = self.weighted_smooth(input_data)
            else:
                filtered_input = input_data
            # update mean and std
            self.update_buffer_mean_std(filtered_input)
            # update buffer
            self.buffer.pop(0)
            self.buffer.append(filtered_input)
        return output_value, output_time
            
    # useful function 
    def mean(self,vector):
        return sum(vector)/len(vector)

    def std(self,vector):
        return (sum([(x-self.mean(vector))**2 for x in vector])/ \
            (len(vector)-1))**(0.5)

    def update_buffer_mean_std(self,x):
        vector = self.buffer[:]+[x]
        # import copy
        # import numpy as np
        # vector = copy.deepcopy(self.buffer)
        # vector.append(x)
        # self.mv_avg = np.mean(vector)
        # self.mv_std = np.std(vector)
        self.mv_avg = self.mean(vector)
        self.mv_std = self.std(vector)

    def weighted_smooth(self,value):
        import numpy as np
        # weight = np.exp(-(value-self.mv_avg)**2)
        # weight = 1/(value-self.mv_avg)
        weight = 0.5
        return weight*value+(1-weight)*self.mv_avg

class peak_detection_zscore:
    """
    using moving window to calculate mean and std and threshold peak by zscore
    """
    def __init__(self,threshold=3,len_win=500,peak_decay_threshold=5,smooth_weight=0):
        self.threshold = threshold
        self.len_win = len_win
        # peak report
        self.peak_decay_threshold=peak_decay_threshold
        self.peak_start = []
        self.peak_end = []
        self.local_peak = []
        self.local_peak_time = []
        # keep track of mean and std
        self.smooth_weight = smooth_weight
        self.mv_avg = -1
        self.mv_std = -1
        # buffer for moving window
        self.buffer = []
        self.buffer_time = []

    # peak detection
    def peak_detect(self, input_data, input_time):
        output_value = 0
        output_time = 0
        # initialize buffer
        if not self.buffer:
            self.buffer.append(input_data)
            self.buffer_time.append(input_time)
            self.mv_avg = input_data
            self.mv_std = 0
            return output_value, output_time
        # calculate mean and std
        if len(self.buffer)<self.len_win:
            self.update_buffer_mean_std(input_data)
            self.buffer.append(input_data)
            self.buffer_time.append(input_time)
        else:
            # compare amplitude

            if abs(input_data-self.mv_avg) > self.threshold*self.mv_std:
                print('enter threshold')
                if not self.peak_start:
                    self.peak_start = input_time
                    self.local_peak = abs(input_data-self.mv_avg)
                    self.local_peak_time = input_time
                elif abs(input_data-self.mv_avg) >= self.local_peak:
                    # print('enter peak record')
                    self.local_peak = abs(input_data-self.mv_avg)
                    self.local_peak_time = input_time
                else:
                    # print('enter downhill')
                    # track if peak goes down for more than 5 seconds
                    if input_time - self.local_peak_time > self.peak_decay_threshold and \
                       self.local_peak_time - self.peak_start > self.peak_decay_threshold:
                        # print('enter report')
                        output_value = self.local_peak
                        output_time = self.local_peak_time
                        # reset local peak
                        self.peak_start = []
                # update buffer
                filtered_input = self.weighted_smooth(input_data,self.buffer[-1],self.smooth_weight)
            else:
                filtered_input = input_data
            # update mean and std
            self.update_mv_mean_std(filtered_input)
            # update buffer
            self.buffer.pop(0)
            self.buffer.append(filtered_input)
        return output_value, output_time
            
    # useful function 
    def mean(self,vector):
        return sum(vector)/len(vector)

    def std(self,vector):
        return (sum([(x-self.mean(vector))**2 for x in vector])/ \
            (len(vector)-1))**(0.5)
    
    def update_mv_mean_std(self, x):        
        self.mv_avg = self.mv_avg+(x-self.buffer[0])/self.len_win
        # a = self.mv_std**2
        # b = (x-self.buffer[0])**2/(self.len_win*(self.len_win-1))
        # c = ((x-self.mv_avg)**2-(self.buffer[0]-self.mv_avg)**2)/(self.len_win-1)
        # tmp_var = self.mv_std**2 + \
        #           (x-self.buffer[0])**2/(self.len_win*(self.len_win-1)) + \
        #           ((x-self.mv_avg)**2-(self.buffer[0]-self.mv_avg)**2)/(self.len_win-1)
        # self.mv_std = tmp_var**(0.5)
    
    def update_buffer_mean_std(self,x):
        old_mean = self.mv_avg
        self.mv_avg = (self.mv_avg*len(self.buffer)+x)/(len(self.buffer)+1)
        tmp_var = (len(self.buffer)-1)*self.mv_std**2 + \
                  (len(self.buffer)*(old_mean-self.mv_avg)**2) + \
                  (x-self.mv_avg)**2
        self.mv_std = (tmp_var/len(self.buffer))**(0.5)

    def weighted_smooth(self,value,previous,weight):
        return weight*value+(1-weight)*previous

    # to do list (intead of using weighted, use gaussian)
    def kernel_smooth(self,value,mu,sigma):
        import numpy as np
        return  value*np.exp(np.power(value-mu,2)/(2*sigma**2))




