
# coding: utf-8

# # Making Sense of the Code
# Goal: to understand and clear up some code that Chris wrote for the actiwatche analysis.

# ## bin_extractor.py
# Extracts values __from binary file__ and outputs it __to a csv__ file with the data __averaged over each minute__.

# ### Important Note for Hex
# From the R package `GENEAread`, I found the function that is used to convert the hexstream to extract the x,y,z values.
# 
# The input to this function is a list of all characters in the hex string of a block (block refering to the repeating structure of the bin file which starts with _Recorded Data_.
# 1. Convert each hex character to its base 10 equivalent ( `[int(x,16) for x in stream]`)
# 2. Split the list (of now ints) into groups of 3s.
# 3. Does a bitshift operation on subgroups:
#     - 1st element out of 3: x<<8
#     - 2nd element out of 3: x<<4
#     - 3rd element out of 3: x
# 4. Sum all elements by group of 3s and put into a new list
# 5. For all elements >=2048, do -2 * maxint + x
# ```python
# [ (-2*2048 + x) if x>=2048 else x for x in list]
# ```
# 6. In this corrected list, create groups of 4
#     - 1st to 3rd encode x,y,z
#     - 4th element encodes the light (>>2) and status of button(binary, if the 2nd bit is 1 then button true else button false)

# ```R
# #internal function for read.bin
# convert.hexstream <-function (stream) 
# {
# 
# maxint <- 2^(12 - 1)
# 
# #packet <- as.integer(paste("0x",stream,sep = "")) #strtoi is faster
# packet <-bitShiftL(strtoi(stream, 16),4*(2:0))
# packet<-rowSums(matrix(packet,ncol=3,byrow=TRUE))
# 
# packet[packet>=maxint] <- -(maxint - (packet[packet>=maxint] - maxint))
# 
# packet<-matrix(packet,nrow=4)
# 
# light <- bitShiftR(packet[4,],2)
# button <-bitShiftR(bitAnd(packet[4,],2),1)
# 
# packet<-rbind(packet[1:3,],light,button)
# 
# packet
# }
# ```

# In[19]:


import logging
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd
import re
from datetime import timedelta



def get_basic_device_info(data):
    """
    Takes in list of lines from binary file and returns a dictionary containing calibration information (gain,offset,volts,lux) and frequency of device
    """
        
    try:
        start_cal_index=data.index("Calibration Data")+1
        end_cal_index=data.index("",start_cal_index)
        calibration_data=data[start_cal_index:end_cal_index]
    except ValueError as e:
        #Can't find Calibration section in binary file
        print(e)
        
    # Replace all spaces in calibration_data by underscores
    # ex of "x gain:2131" --> {'x_gain':2131}
    calibration_data=[i.replace(" ","_").lower() for i in calibration_data]
    calibration_dict={}
    # Fill dictionnary with appropriate values
    for line in calibration_data:
        index_split=line.index(":")
        key,val = line[:index_split],line[index_split+1:]
        calibration_dict[key]=float(val)
            
    # Add frequency measurement
    p=re.compile(r"^Measurement Frequency:(\d[\d.]+)\s+Hz",re.I)
    for i in data:
        tt=p.search(i)
        if tt:
            freq=float(tt.group(1))
            break
    calibration_dict["freq"]=freq
    
    # Calibration dict contains: x gain','x_offset','y_gain','y_offset','z_gain','z_offset','volts','lux','freq'
    return calibration_dict

def parse_one_block(recording_block_list):
    """
    Given a block of recording (a page), extract start time, temperate, hexstring and return a df (dataframe)
    """
    temperature=None
    page_time=None
    hexstring=None
    
    # Debug: to see Sequence Number
#     print(recording_block_list[2])
#     print("Length of one-block list: %d"%len(recording_block_list))
    
    for index in range(len(recording_block_list)):
        tmp=recording_block_list[index]
        
        if tmp.startswith("Page Time"):
            if page_time==None:
                index_start_t=tmp.index(":")+1
                time_str=tmp[index_start_t:].strip()
                page_time=pd.to_datetime(time_str,format="%Y-%m-%d %H:%M:%S:%f")
                
        elif tmp.startswith("Temperature"):
            if temperature==None:
                temperature=float(tmp.split(":")[1])
                                         
        elif tmp.startswith("Measurement Frequency"):
            if hexstring==None:
                # Add one to the index, the hexstring is the one after Measurement Frequency
                hexstring=recording_block_list[index+1].strip()
                break
                
    # If it can't find either of the variables
    if temperature==None or page_time==None or hexstring==None:
        return
        
    # Turn the hexstring into useful data (ie get x,y,z and light info out of it)
    df= hex_to_int_df(hexstring,page_time)
    
    # For all these points in time, we consider the temperature to be the same (add temp column)
    df["temperature"]=temperature
    
    return df
    
    
def hex_to_int_df(hexstream,start_timestamp):
    """
    Converts each hex character (base 16) to an int and returns x,y,z matrix
    """
    #Convert each char (hex) to int
    intstream=[int(x,16) for x in hexstream]
    
    #Group elements by 3, do bit shifting operations to extract meaning in ints and then sum all elements in groups of 3s to get x,y,z and light values 
    meaningful_ints=[]
    for i in range(0,len(intstream),3):
        pos0=intstream[i]<<8
        pos1=intstream[i+1]<<4
        pos2=intstream[i+2]
        meaningful_ints.append(pos0+pos1+pos2)
    
    # All elements at or above 2**11=2048 are meant to be negative and so need to be converted with formula (-2*2048 +x)
    meaningful_ints=[-4096+x if x>=2048 else x for x in meaningful_ints]
    
    # Make a matrix where the columns are x,y,z,light and the rows are the instances
    shaped_matrix=np.matrix(meaningful_ints).reshape(len(meaningful_ints)//4,4)
    
    # Adjust 4th column which encode light and button, we only want light
    shaped_matrix[:,3]=shaped_matrix[:,3]>>2
    
    # Format of the timestamp: 2016-06-02 09:04:29:500
    n_row_matrix=shaped_matrix.shape[0]
    start_timestamp=pd.to_datetime(start_timestamp,format="%Y-%m-%d %H:%M:%S:%f")
    time_index=pd.date_range(start=start_timestamp,periods=n_row_matrix,freq=".1S")
    df=pd.DataFrame(shaped_matrix,columns=["x","y","z","light"],index=time_index)
    
    # Though the dataframe has a time index and column names
    # Note that the values have not been calibrated yet though
    return df

def calibrate_df(df,calibration_dict):
    """
    Takes the dataframe with the x,y,z,light and temperature column.
    Using the calibration dict, it adjusts the xyz columns
    """
    
    df['x']=(df['x']*100-calibration_dict["x_offset"])/calibration_dict["x_gain"]
    df['y']=(df['y']*100-calibration_dict["y_offset"])/calibration_dict["y_gain"]
    df['z']=(df['z']*100-calibration_dict["z_offset"])/calibration_dict["z_gain"]
    df['light']=df['light']*calibration_dict['lux']/calibration_dict['volts']
    
    return df
    
def calc_block_start_and_step(ref_string,recording_data):
    """
    Calculate the step between each block (page) of recorded data as well as in the index for the start of the recording blocks.
    """
    index1=recording_data.index(ref_string)
    index2=recording_data.index(ref_string,index1+1)
    return index1, index2-index1

def combine_all_data_to_df(path_binary_file):
    """
    Find all blocks of recorded data, send them to be parsed and combine them all afterwards.
    Multiprocessing elegible.
    """
    with open(path_binary_file,'r') as f:
        data = [line.strip() for line in f]
        
    
    # Get information required to split data in blocks
    ## Get index of the start of the blocks 
    block_start, step = calc_block_start_and_step("Recorded Data", data)
    ## List containing multiple pairs of block + calibration data
    tmp_data=data[block_start:]
    blocks_data=[tmp_data[i:i+step] for i in range(0,len(tmp_data),step)]
    print("Length of subdivided list: %d"%len(blocks_data))
    
    with Pool(multiprocessing.cpu_count()-1) as p:
        r=p.map_async(parse_one_block,blocks_data)
        r.wait()
    returned_dfs=r.get()
   
    # Concatenate all the small dfs from the individual block parsing
    df=pd.concat(returned_dfs)

    # Calibrate x,y,z and light data
    calibration_dict=get_basic_device_info(data)
    calibrated_df=calibrate_df(df,calibration_dict)
    
    return calibrated_df
    


# ## Secondary Functions

# - To bin by minutes or another rule and calculate  mean and std
# 
# Might not work though, the reuse of the reduced_df wouldn't work
# 
# More importantly, **might not need this**

# In[3]:


def reduce_df(df,reduce_rule_minute=1):
    """
    Takes in a df with index (going up to ms) and reduces it into indexes representing each minute or another time frame ('5S'). Calculate the mean and std for all columns (x,y,z,light)
    """
    # Group df by default to reduce_rule (T == 1 minute bins)
    reduce_rule="{:.2g}T".format(reduce_rule_minute)
    reduced_df=df.resample(reduce_rule)
    col_names=df.columns
    op_names=["mean","std"]
    
    new_columns={}
    
    # Create the new columns
    for op in op_names:
        for col in col_names:
            new_col= col+"_"+op
            if op=="mean":
                new_columns[new_col]=reduce_df[col].mean()
            elif op=="std":
                new_columns[new_col]=reduce_df[col].std()
        
    # Drop the old columns, only keep means and stds
    reduce_df=reduce_df.drop(col_names,axis=1)
    
    return reduced_df


# Find by how much to shift the df (since the rolling groups on the last value's position/ ie to the right rather than to the left)

# In[23]:


def calc_angle(df):
    """
    Calculate the angle given the x, y and z values.
    Equation:
        atan(z/sqrt(x^2 +y^2)) *180/pi
    """
    necessary_cols=['x','y','z']
    for elem in necessary_cols:
        if not elem in df.columns:
            return None
    
    angle=df['z']/np.sqrt(np.power(df['x'],2)+np.power(df['y'],2))
    angle=np.arctan(angle)*180/np.pi
    return angle


# In[4]:


def find_datetime_shift(self,seconds=5):
    shift_val=1
    delta_t=timedelta(seconds=seconds)
    while True:
        if (self.df.index[shift_val]-self.df.index[0])>=delta_t:
            return shift_val-1
        elif shift_val>10**4:
            break
        else:
            shift_val+=1


# In[66]:


def roll_window(df,operation='median',shift=True,window_seconds=5):
    """
    Takes a dataframe and creates rolling windows. Using those
    windows, it computes either the median, mean or std of each window.
    """
    if window_seconds>0:
        roll_seconds="{}S".format(window_seconds)
    else:
        return None
    
    # If not dataframe
    if not isinstance(df,pd.DataFrame):
        return None
        
    df=df.rolling(roll_seconds)
    
    # Apply operation
    if operation=='median':
        df=df.median()
    elif operation=='mean':
        df=df.mean()
    elif operation=='std':
        df=rdf.std()
    else:
        return None
    
    # If you want the windows to be calculated on the left (rather than the right)
    if shift:
        shift_val=-1*self.find_datetime_shift(complete_df,seconds=window_seconds)
        df=df.shift(shift_val)
    
    return df


# In[62]:


def determine_activity(df,window_minutes=5,diff_angle_threshold=5):
    """
    Computes the change of angles between each point (avg of 5seconds)
    and then rolls on windows of size window_minutes to determine if
    the max change is bigger than angles_threshold or if the -1 * min
    change is bigger than angles_threshold. Inactivity is when the 
    values are between -threshold and +threshold.
    """
    # Checking as to the type of input
    if isinstance(df,pd.DataFrame) :
        if 'angles' in df.columns:
            angles=df['angles']
        elif 'angle' in df.columns:
            angles=df['angle']
        else:
            return None
    elif isinstance(df,pd.Series):
        angles=df
    else:
        return None
    
    # Rolling rule prep
    if (isinstance(window_minutes, int) or         isinstance(window_minutes,float)) and window_minutes>0:
        roll_rule="{:.2g}T".format(window_minutes)
        logging.info("'determine_activity': window for the roll=  {} minutes".format(roll_rule))
    else:
        loggin.error("'determine_activity': window_minutes is not a valid input [{}]".format(window_minutes))
        return None
    
    # Absolute value of change between angle
    diff_angles=angles.diff().abs()
    
    def window_inactivity(window):
        """
        Given a window in a roll, determine if the maximum diff 
        in the window is lower than 5deg.
        Inactivity == 1
        Activity == 0
        """
        absolute_max=window.max()
        if absolute_max<diff_angle_threshold:
            return 1
        else:
            return 0
    
    inactivity=diff_angles.rolling(roll_rule).apply(window_inactivity)
    return inactivity


# In[ ]:


def compress_windows_df(data,c_w_minutes=0,c_w_seconds=0,operation='sum'):
    """
    Takes a dataframe/serie, groups tries to group by `c_w_minutes` if not
    with `c_w_seconds` and else it fails.
    Then after resampling the data, it applies `operation` on it.
    `operation`: sum, mean, std, median, count
    """
    if c_w_minutes>0:
        # Convert compress_window_minute to appropriate format
        sampling_rule="{:.2g}T".format(c_w_minutes)
    elif c_w_seconds>0:
        # Convert compress_window_minute to appropriate format
        sampling_rule="{:.2g}S".format(c_w_seconds)
    else:
        return None
    
    resampled_data=data.resample(sampling_rule)
    
    if operation == 'sum':
        resampled_data=resampled_data.sum()
    elif operation == 'mean':
        resampled_data=resampled_data.mean()
    elif operation == 'std':
        resampled_data=resampled_data.std()
    elif operation == 'median':
        resampled_data=resampled_data.median()
    elif operation == 'count':
        resampled_data=resampled_data.count()
    else:
        logging.error("'compress_data': operation '{}' not supported.".format(operation))
        return None
        
    return resampled_data


# In[ ]:


def to_csv(df,output_directory,patient_code,step_name,time_format="%Y-%m-%d %H:%M:%S"):
    """
    Given a dataframe, save it with the name of the step.
    Within the output_directory, have directories for 
    each participant and within them the different 
    intermediate steps and the final output.
    """
    # Type check of df
    if not (isinstance(df,pd.DataFrame) or isinstance(df,pd.Series)):
        logging.error("The input is not an instance of a DataFrame or a Series [type={}]".format(str(type(df))))
        return None
    
    # If patient_code folder does not exist create it
    patient_code=patient_code.strip().replace(" ","_")
    patient_path=os.path.join(output_directory,patient_code)
    
    # Create directory if it does not exist
    if not os.path.exists(patient_path):
        os.mkdir(patient_path)
    
    file_name=patient_code+"__"+step_name+".csv"
    file_path=os.path.join(patient_path,file_name)
    df.to_csv(file_path,date_format=time_format)
        
    


# #### In The Works

# In[68]:


def merge_tables(df1,df2,axis=1):
    """
    Either adds columns or rows.
    Axis=1 means to add columns.
     - Checks the index of both, determines if equal
     - Adds columns in df2 that are not in df1
    Axis=0 means to add rows.
     - Simpler append/concat
    """
    
    # Adding rows (ex 2 files of one patient)
    if axis==0:
        ## Make sure that both dataframes have an index that is of type DatetimeIndex
        if not (isinstance(df1,pd.DatetimeIndex) and isinstance(df2,pd.DatetimeIndex)):
            logging.warning("`merge_tables`: one of the data structure passed does not have an datetime as the index.")
            return None
        
        ## Determine which file comes first
        if df1.index[0]>df2.index[0]:
            first=df2
            second=df1
        elif df1.index[0]<df2.index[0]:
            first=df1
            second=df2
        else:
            logging.warning("`merge_tables`: both dataframes start at the same timepoint.")
            return None
        
        ## Determine if there is overlap in the beginning and end of the recordings
        if first.index[-1]>second.index[0]:
            # Overlap, apply function that determines action (with option to change default behaviour)
            first,second=overlap_action(first,second,behaviour="find")
        appended_df=pd.concat([first,second],axis=0,join='outer',ignore_index=False)
        return appended_df
    # Adding columns 
    elif axis ==1:
        all_columns=pd.concat([first,second],axis=1,join='inner')
        return all_columns
    else:
        logging.warning("`merge_tables`: axix option is invalid; for the addition of rows → axis=0, for the addition of columns → axis=1.")
        return None
        
    


# In[99]:


def overlap_action(first,second,behaviour="find"):
    """
    Deal with overlapping data.
    Need to find intersect (section of interest).
    Apply given behaviour.
    """
    # Get index of intersect
    intersect_end_first=first.index.intersection(second.index)
    intersect_begin_second=second.index.intersection(first.index)
    
    if behaviour=='find':
        intersect_info={'first':{'start':intersect_end_first[0],'end':intersect_end_first[-1]},'second':{'start':intersect_begin_second[0],'end':intersect_begin_second[-1]}}
        return intersect_info
    elif behaviour==''
    
    # Start looking from the end of the intersect_end_first 
    # and from the beginning of the intersect_begin_second
    return first,second


# ### Determine on/off

# In[139]:


diff_angle=angle.diff()


# In[135]:


diff_angle.head()


# In[140]:


diff_angle=diff_angle.resample('5T').std()


# In[142]:


diff_angle.head()


# In[141]:


diff_angle.isnull()


# In[149]:


(diff_angle<0.05).sum()


# In[150]:


low_diff= (diff_angle<0.05)


# In[129]:


diff_angle.shape


# In[151]:


low_diff.sum()


# In[155]:


low_diff_index=low_diff[low_diff].index.tolist()


# In[156]:


for i in range(100):
    print(low_diff_index[i])


# ### Merge tables Tests

# In[97]:


angle.describe()


# In[92]:


angle.index.intersection(complete_df.index[-10:])


# In[93]:


get_ipython().run_line_magic('pinfo', 'pd.concat')


# In[87]:


angle.index[-1]


# In[78]:


isinstance(complete_df.index,pd.DatetimeIndex)


# In[70]:


angle.index[0]


# In[72]:


complete_df.index[1]


# In[74]:


angle.index[1]>complete_df.index[0]


# ### Rolie Polie Olie Tests
# Testing rolling dataframe and series

# In[65]:


get_ipython().set_next_input("angle[:200].resample('2S').min");get_ipython().run_line_magic('pinfo', 'min')


# In[ ]:


angle[:200].resample('2S').min


# In[59]:


diff_angle=angle.diff().abs()
diff_angle[:10]


# In[60]:


diff_angle[:10].rolling('0.2S').count()


# In[64]:


diff_angle[:100].rolling('0.2T').apply(window_inactivity,kwargs={'diff_angle_threshold':0.2}).head(10)


# In[251]:


angle.diff().abs().head()


# In[262]:


angle.head(100).diff()


# In[9]:


def angle_diff_resample(angles,resample_rule='5S'):
    return angles.resample(resample_rule).agg({'abs_diff': lambda x: x.max()-x.min()})

def activity_eval(angles,cutoff=5,resample_rule="T"):
    """
    Given a series of angles, calculate a series of true/falses and
    sum them in resampled groups of ex 1 minute, 10 minutes ...
    """
    act_inact=angles > cutoff
    return act_inact.resample(resample_rule).sum()


# ### Testing my fncs
# 
# Yeeeah, it seems my funcs return the same values as the R package and it runs decently fast (start to calibrated df is about 45 sec / per binary file)

# In[10]:


path_binary_file="../Data/10__031957_2016-06-17 11-15-20.bin"
with open(path_binary_file) as f:
    data=[line.strip() for line in f]


# In[12]:


from time import time


# In[13]:


t0=time()
complete_df=combine_all_data_to_df(path_binary_file)
t1=time()
print("It took %d seconds"%(int(t1-t0)))


# In[184]:


# t2=time()
# for col in complete_df.columns:
#     complete_df[col][:100].rolling('5s').median()
# t3=time()
# aa=complete_df[:100].rolling('5s')
# aa=aa.median()
# t4=time()
# print("Rolling one col takes %f" %(t3-t2))
# print("Rolling the whole df takes %f " %(t4-t3))


# In[221]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
complete_df.x.resample('10T').mean().plot(ax=ax)
fig.savefig("original x's.png")


# In[212]:


complete_df.x.rolling('5s').median


# In[21]:


test_roll=roll_window(complete_df)


# In[24]:


angle=calc_angle(test_roll)


# In[25]:


angle.shape


# In[31]:


angle.rolling(100).min().shape


# In[26]:


angle.resample('5T').min().shape


# In[32]:


angle[:10]


# In[45]:


angle[:200].rolling("0.01S").max().head(20)


# In[43]:


angle[:200].rolling("0.005T").max().head(20)


# In[29]:


angle[:20].diff().abs()


# In[244]:


angle.resample('5S').max().head()


# In[241]:


angle.resample('5S').agg({'abs_diff': lambda x: x.max()-x.min()}).head()


# In[220]:


fig, ax = plt.subplots()
test_roll.x.resample('10T').mean().plot(ax=ax)
fig.savefig("rolled windows x's.png")


# In[223]:


(test_roll.x==complete_df.x).sum()


# In[225]:


1/complete_df.shape[0]*1417805


# In[6]:


complete_df.shape


# In[9]:


complete_df.index


# In[18]:


complete_df.x.iloc[:5]


# In[27]:


complete_df.x.iloc[:50].median()


# In[123]:


complete_df.head()


# In[32]:


complete_df.x.iloc[:50].rolling('5s').median()[49]


# In[33]:


complete_df.x[:10]


# In[133]:


complete_df.rolling('5s').sum()


# In[88]:


from datetime import timedelta
def find_datetime_shift(df,seconds=5):
    shift_val=1
    delta_t=timedelta(seconds=seconds)
    while True:
        if df.index[shift_val]-df.index[0]>=delta_t:
            return shift_val-1
        elif shift_val>10**4:
            break
        else:
            shift_val+=1
    


# In[68]:


find_datetime_shift(complete_df.x)


# In[115]:


# Do the rolling median with a window of 'windows_seconds' seconds
t0=time()

window_seconds=5
shift_val=-1*find_datetime_shift(complete_df,seconds=window_seconds)
roll_seconds="%ds"%(window_seconds)
res={}
for col in complete_df.columns:
    print(col)
    tmp=complete_df[col].rolling(roll_seconds)#.median()
#     tmp=tmp.shift(shift_val)
    res['rolled_%s'%col]=tmp

t1=time()
print("It took %d seconds to roll and shift."%(t1-t0))


# In[120]:


complete_df.median()


# In[111]:


import sys
get_ipython().run_line_magic('pinfo', 'sys.getsizeof')


# In[134]:


empty=pd.DataFrame(index=complete_df.index)


# In[137]:


empty=pd.concat([empty,complete_df.y],axis=1)


# In[116]:


sys.getsizeof(pd.DataFrame(res))/1000/1000 #MB


# In[108]:


complete_df.head(10)


# In[98]:


rolled_x.head()


# In[89]:


find_datetime_shift(complete_df,seconds=0.3)


# In[81]:


complete_df.x[:5]


# In[90]:


complete_df.x[:5].rolling('300ms').median().shift(-1*find_datetime_shift(complete_df,0.3))


# In[35]:


complete_df.x[:10].shift(-1)


# In[ ]:


for i in complete_df.columns:
    print(i)


# In[ ]:


#Number of minutes in recording
complete_df.shape[0]/10/60


# In[ ]:


#Number of days
complete_df.shape[0]/10/60/60/24


# In[ ]:


complete_df.iloc[999]


# In[ ]:


complete_df.resample("T").mean().head()


# In[ ]:


complete_df.x.head()


# In[ ]:


# Resample by minute: T
res=complete_df.resample("T").std()
res['x_mean']=res.x.mean()


# In[ ]:


res['x_std']=res.x.std()


# In[ ]:


res.drop(complete_df.columns,axis=1)


# In[ ]:


data.index("Calibration Data")
data[45:60]


# In[ ]:


def test_pool_2args(arg1,arg2):
    print("-"*20)
    print(arg1)
    print()
    print(arg2)

exdict={"a":1,"b":2}
pairedList=[(data[i:i+10],exdict) for i in range(59,100,10)]
#with Pool(4) as p:
#    p.map(test_pool_2args,data[:10],exdict)
for i in pairedList:
    print(i)
    print("-"*20)


# In[ ]:


multiprocessing.cpu_count()


# ## Original Code w/ some modifications and annotations
# **Split code between functions and execution.**
# 
# Imports

# In[ ]:


# %load ../OGCode/bin_extractor
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import time
import matplotlib.pyplot as plt
import os
import math


# ### Function definitions

# Function: `get_SVM` (sum vector matrix?)
# 
# It seems like a sum of the eucledian distances of the angles, but why the -1?

# In[ ]:


def get_SVM(xvec, yvec, zvec):
    svm = 0
    for xval, yval, zval in zip(xvec, yvec, zvec):
        svm += abs(math.sqrt(xval**2 + yval**2 + zval**2) -1)
    return svm


# Function: `get_base_values`
# 1. Open binary file and read line by line into a list
# 2. Find calibration data from said list
# 3. Find measurement frequency
# 4. Split list to only include recordings
# 5. Iterate over that list to obtain a list of all the temperatures, a list of all the timestamps and a long stiched string of the hexstream.
# 
# Could clean up by:
# - utilizing a dictionnary for the configuration and calibration information of the watch.
# - returning a dataframe for the timestamps, temperatures (and possibly the hexstring)
# - Size of hexstream in memory is almost the size of the file (150MB string and 159MB file)
#     - By having more small chunks could use a multithreading option to parse it afterwards
# - loop through recording_data once and get what you need at each level rather than go through it multiple times for each variable
# - read files into lists like so (making sure file is closed): 
# ```python
# with open(fn,'r') as f:
#     data=[line.strip() for line in f]
# ```
# 
# Why is he only passing a string for the timestamps? Wouldn't it be more useful to convert that to a timestamp object.

# In[ ]:


def get_base_values(binpath):
    ## Get raw values, calibration data
    binfile_data = [val.strip() for val in open(binpath).readlines()]
    
    ## Get Calibration data from binary file
    #calibration_data = binfile_data[47:55]
    try:
        start_cal_index=binfile_data.index("Calibration Data")+1
        end_cal_index=binfile_data.index("",start_cal_index)
        calibration_data=binfile_data[start_cal_index:end_cal_index]
    except ValueError as e:
        #Can't find Calibration section in binary file
        #Deal with error
        print(e)
        ## Using a dict here would make things nicer
    x_gain, x_offset, y_gain, y_offset, z_gain, z_offset, volts, lux = [int(val.split(':')[1]) for val in calibration_data]
    
    ## Get measurement frequency from binary file
    #freq = float(binfile_data[19].split(":")[1].split()[0])
    p=re.compile(r"^Measurement Frequency:(\d[\d.]+)\s+Hz",re.I)
    for i in binfile_data:
        tt=p.search(i)
        if tt:
            freq=float(tt.group(1))
            break
    
    ## Split list to get recording data
    start_recording_index=binfile_data.index("Recorded Data")
    recording_data = binfile_data[start_recording_index:]
    print("Size of the recording_data list: %.3fMB"%(sys.getsizeof(recording_data)/10**6))
    
    
    ### Variables
    def get_index_first(match_string,list_data):
        p=re.compile(match_string,re.I)
        for index,line in enumerate(list_data):
            if p.search(line):
                return index
        return None
    
    
    ## To be sure of the step, calculate diff between to first index
    ## for extra added security, could make sure that the sublist has the same format (slows things down though)
    def calc_step(ref_string,recording_data):
        index1=recording_data.index(ref_string)
        index2=recording_data.index(ref_string,index1+1)
        return index2-index1
    
    step_btw_recs=calc_step("Recorded Data",recording_data)
    len_data=len(recording_data)

    ## Get a list of all the temperatures and convert to float
    ### Assumption: temperature @ index+ x*step_btw_recs (x:0,1,2,...)
    index_temp_start=get_index_first(r"Temperature",recording_data)
    temp = [float(val.split(':')[1]) for val in recording_data[index_temp_start:len_data:step_btw_recs]]
    
    ## Get a list of all the timestamps
    ### Assumption: timestamps @ index+ x*step_btw_recs (x:0,1,2,...)
    index_timestamp_start=get_index_first(r"Page Time",recording_data)
    timestamps = [val[val.index(':')+1:] for val in recording_data[index_timestamp_start:len_data:step_btw_recs]]

    # Get hexstreams from 8th line of each recording
    index_hex_start=get_index_first(r"Measurement Frequency",recording_data)+1
    hexstream = "".join([recording_data[i] for i in range(index_hex_start,len_data,step_btw_recs)])
    
    return timestamps, temp, hexstream, freq, x_gain, x_offset, y_gain, y_offset, z_gain, z_offset, volts, lux


# Just testing out some stuff

# In[ ]:


import re
test_binary_file="../Data/10__031957_2016-06-17 11-15-20.bin"
binfile_data = [val.strip() for val in open(test_binary_file).readlines()]
p=re.compile(r'^Measurement Frequency:(\d+(.\d+)?)\s+Hz',re.I)
for i in binfile_data:
    tt=p.search(i)
    if tt:
        freq=tt.group(1)
        break
        
print(freq)



# In[ ]:


tmp=get_base_values(test_binary_file)


# Checking the distribution of the hexstring

# In[ ]:


def get_distribution_hex(hexstring,length_block=12):
    p={}
    ll=len(hexstring)
    print("Length string: %d"%ll)
    n_blocks=ll/length_block
    print("Number blocks: %d"%n_blocks)
    for i in range(length_block):         
        p[i]={}         
        for x in range(0,ll,12):
            cur=hexstring[i+x]
            p[i][cur]=p[i].get(cur,0)+1
        for key in p[i].keys():
            p[i][key]=p[i][key]/n_blocks
    return p

hexdistribution=get_distribution_hex(tmp[2],12)


# In[ ]:


for topkey in hexdistribution:
    print("="*20)
    print(topkey)
    print(hexdistribution[topkey])


# Function: `get_calibrated_values`
# 
# Note:
# - **In the original library in R, there is a bit shift done to the ints**, none seen here. Still left to be determined if necessary or not though.
# - Each line of hex is 3600 char
#     - time between each recorded data== 30secs
#     $$f_{writing\;data}=10Hz \rightarrow n_{char/write}=\frac{3600char}{\frac{30s}{10write/s}}=12char/write$$
# - Hex strings should be subdivided into blocks of 12char
#     - char #9 of each block seems to always be $0$
# 
# Optimization:
# - __Matrices__ are better for these sorts of calcs than lists

# In[ ]:


def get_calibrated_values(hexstream, calibration_values):
    x_gain, x_offset, y_gain, y_offset, z_gain, z_offset, volts, lux = calibration_values
    
    #Converts hex stream to an integer stream
    # Each hex character is 4 bits, so a group of 3 hex characters is 12 bits
    # The integer values are given in two's complement representations: values greater than 2047 (largest 11 bit number) represent negative numbers
    intstream = [val if val<=2048 else -2*2048 + val for val in [int(hexstream[i:i+3], 16) for i in range(0, len(hexstream), 3)]]

    # Calibrate raw values using offsets and gains
    # Note: the calculated values here are slightly different from the R/GENEactiv output because of differences in floating point precision
    raw_x = intstream[0::4]
    raw_y = intstream[1::4]
    raw_z = intstream[2::4]
    
    x_vals = [(val*100-x_offset)/x_gain for val in raw_x]
    y_vals = [(val*100-y_offset)/y_gain for val in raw_y]
    z_vals = [(val*100-z_offset)/z_gain for val in raw_z]
    light = [ (x>>2)*lux/volts for x in intstream[3::4]]
    
    calibrated_values = x_vals, y_vals, z_vals, light
    return calibrated_values


# Function: `get_average_data`

# In[ ]:


#Generates epoch compressed file
def get_average_data(timestamps, temp, calibrated_values):
    x_vals, y_vals, z_vals, light = calibrated_values
    
    avg_x = [sum(x_vals[i:i+600])/600 for i in list(range(len(x_vals)))[0::600]]
    avg_y = [sum(y_vals[i:i+600])/600 for i in list(range(len(y_vals)))[0::600]]
    avg_z = [sum(z_vals[i:i+600])/600 for i in list(range(len(z_vals)))[0::600]]
    avg_light = [sum(light[i:i+600])/600 for i in list(range(len(light)))[0::600]]
    
    x_dev = [np.std(x_vals[i:i+600]) for i in list(range(len(x_vals)))[0::600]]
    y_dev = [np.std(y_vals[i:i+600]) for i in list(range(len(y_vals)))[0::600]]
    z_dev = [np.std(z_vals[i:i+600]) for i in list(range(len(z_vals)))[0::600]]
    SVM = [get_SVM(x_vals[i:i+600], y_vals[i:i+600], z_vals[i:i+600]) for i in list(range(len(z_vals)))[0::600]]

    # Length of temp might be 1 shorter if number of pages is odd
    temp = temp + [0]*(len(z_dev)-len(temp))
    # Add values to pandas data frame
    names = ["time", "X", "Y", "Z", "temp", "X_dev", "Y_dev", "Z_dev", 'SVM', 'light']
    watch_raw_data = pd.DataFrame(dict(zip(names, [timestamps, avg_x, avg_y, avg_z, temp, x_dev, y_dev, z_dev, SVM, avg_light])))
    
    return watch_raw_data


# Function: `vcomp`
# 
# Computes the mean of values in a vector in chunks.
# Bins into a window, calcs avg and returns reduced values.
# 
# The only time this is called in this module is for the list of temperatures.

# In[ ]:


#Returns a compressed vector with averages across successive windows of specified size
def vcomp(vec, size):
    output = []
    for i in range(0, len(vec), size):
        temp = vec[i:i+size]
        if len(temp)<size:
            break
        mean = sum(temp)/len(temp)
        output.append(mean)
    return output


# Function: `extract_bin`
# 1. Calls fnc `get_base_values` and gets 12 values back
#     * Timestamps (list)
#     * Temperatures (list)
#     * Hexstream (string)
#     * Calibration data (string)
#         - freq, x_grain, x_offset, ...
# 2. Return if `freq != 10` (because anything else means the watch didn't work...?)
# 3. Calls fnc `get_calibrated_values` and gets calibrated values back
# 
# Optimization:
# - Using a __dict for the calibration data__ makes it definitly cleaner (mod in `get_base_values` fnc)

# In[ ]:


def extract_bin(binpath):
    #Extract values (30 s epoch timestamps, 30 s epoch temperature, 0.1 s epoch x,y,z values)
    base_values = get_base_values(binpath)
    timestamps, temp, hexstream, freq, x_gain, x_offset, y_gain, y_offset, z_gain, z_offset, volts, lux = base_values
    if freq!=10:
        return
    calibration_values = [x_gain, x_offset, y_gain, y_offset, z_gain, z_offset, volts, lux]
    calibrated_values = get_calibrated_values(hexstream, calibration_values)    
    
    return [calibrated_values, timestamps, temp, freq]


# Function: `epoch_compress`

# In[ ]:


#Makes 60 second epoch compressed file
def epoch_compress(calibrated_values, timestamps, temp, freq, epoch=60):
    num_pages = int(60/(300*(1/freq)))
    timestamps = timestamps[0::num_pages]
    temp = vcomp(temp, num_pages)
    compressed_table = get_average_data(timestamps, temp, calibrated_values)  
    return compressed_table


# Function: `expand_val`

# In[ ]:


def expand_val(invec, factor):
    output = []
    [output.extend( [val]*factor) for val in invec]
    return output


# ### Main call

# From

# In[ ]:



if __name__ == '__main__':
   #baseroot = "D:/Clement/Python/refaire"
   baseroot="."
   binroot = "200917"
   outroot = "1 min tables"
   raw_table_root = "rawxyz"
   #Seems deprecated
   if not os.path.exists(raw_table_root):
       os.mkdir(raw_table_root)
   
   if not os.path.exists(baseroot + '/' + outroot):
       os.mkdir(baseroot + '/' + outroot)


# To

# In[ ]:


def setup_folders(working_dir_path, output_bin_convert_folder_name ):
    if working_dir_path=='.' or working_dir_path=="":
        working_dir_path=os.getcwd()
    output_path=os.path.join(working_dir_path,output_bin_convert_folder_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path


# Should the pcode file be hardcoded?
# Probably not...

# In[ ]:


bin_to_pcode_file = os.path.join('../OGCode',"bin file to pname.txt")


# In[ ]:


bin_to_pcode_file


# Creates a key, value combination of the binary_filename and the patient_code

# In[ ]:


bin_to_pcode = dict([[x.strip() for x in val.split('\t')] for val in open(bin_to_pcode_file).readlines()[1:]])


# In[ ]:


bin_to_pcode


# Creates an ordered list of the files ending with '.bin' from the `binroot` path.

# In[ ]:


binnames = list(sorted([val for val in os.listdir(binroot) if val.endswith(".bin")]))


# In[ ]:


for binname in binnames
    binpath = binroot + '/' + binname
    outname = bin_to_pcode.get(binname, '') + '.txt'
    if outname == '.txt':
        outname = binname[0:-4] + '.txt'
        
    outpath = baseroot + '/' + outroot + '/' + outname
    if os.path.exists(outpath):
        continue
    
    #Extract values from bin file
    print("Extracting", binname)
    bin_vals = extract_bin(binpath)
    
    if bin_vals is None:
        print(binname, 'frequency is not 10 Hz')
        continue
    else:
        calibrated_values, timestamps, temp, freq = bin_vals
    
    #Output 1 min epoch compressed file
    print("Outputting 1 min epoch compressed file")
    compressed_table = epoch_compress(calibrated_values, timestamps, temp, freq, epoch=60)
    compressed_table.to_csv(outpath, sep='\t', index=False)
    
##        #Output timestamps
##        print("Outputting raw value file")
##        format_str = "%Y-%m-%d %H:%M:%S:%f"
##        startstamp = str(int(datetime.strptime(timestamps[0], format_str).timestamp()))
##        endstamp = str(int(datetime.strptime(timestamps[-1], format_str).timestamp()))
##        outstamps = startstamp + " " + endstamp
##        stamp_outpath =  baseroot + '/' + raw_table_root + '/' + outname[0:-4] + "_timestamps.txt"
##        open(stamp_outpath, 'w').write(outstamps)
##        
##        #Output rawxyz file
##        x_vals, y_vals, z_vals, light = calibrated_values
##        temp = expand_val(temp, 300)
##        
##        output_vals = [x_vals, y_vals, z_vals, temp, light]
##        names = ["X", "Y", "Z", "Temp", "Light"]
##        raw_xyz = pd.DataFrame(dict(zip(names, output_vals)))
##        
##        raw_table_outpath = baseroot + '/' + raw_table_root + '/' + outname
##        raw_xyz.to_csv(raw_table_outpath, index=False, sep='\t', columns = names)


# In[ ]:



binn

:
    binpath = binroot + '/' + binname
    outname = bin_to_pcode.get(binname, '') + '.txt'
    if outname == '.txt':
        outname = binname[0:-4] + '.txt'
        
    outpath = baseroot + '/' + outroot + '/' + outname
    if os.path.exists(outpath):
        continue
    
    #Extract values from bin file
    print("Extracting", binname)
    bin_vals = extract_bin(binpath)
    
    if bin_vals is None:
        print(binname, 'frequency is not 10 Hz')
        continue
    else:
        calibrated_values, timestamps, temp, freq = bin_vals
    
    #Output 1 min epoch compressed file
    print("Outputting 1 min epoch compressed file")
    compressed_table = epoch_compress(calibrated_values, timestamps, temp, freq, epoch=60)
    compressed_table.to_csv(outpath, sep='\t', index=False)
    
##        #Output timestamps
##        print("Outputting raw value file")
##        format_str = "%Y-%m-%d %H:%M:%S:%f"
##        startstamp = str(int(datetime.strptime(timestamps[0], format_str).timestamp()))
##        endstamp = str(int(datetime.strptime(timestamps[-1], format_str).timestamp()))
##        outstamps = startstamp + " " + endstamp
##        stamp_outpath =  baseroot + '/' + raw_table_root + '/' + outname[0:-4] + "_timestamps.txt"
##        open(stamp_outpath, 'w').write(outstamps)
##        
##        #Output rawxyz file
##        x_vals, y_vals, z_vals, light = calibrated_values
##        temp = expand_val(temp, 300)
##        
##        output_vals = [x_vals, y_vals, z_vals, temp, light]
##        names = ["X", "Y", "Z", "Temp", "Light"]
##        raw_xyz = pd.DataFrame(dict(zip(names, output_vals)))
##        
##        raw_table_outpath = baseroot + '/' + raw_table_root + '/' + outname
##        raw_xyz.to_csv(raw_table_outpath, index=False, sep='\t', columns = names)


# In[ ]:


:
        binpath = binroot + '/' + binname
        outname = bin_to_pcode.get(binname, '') + '.txt'
        if outname == '.txt':
            outname = binname[0:-4] + '.txt'
            
        outpath = baseroot + '/' + outroot + '/' + outname
        if os.path.exists(outpath):
            continue
        
        #Extract values from bin file
        print("Extracting", binname)
        bin_vals = extract_bin(binpath)
        
        if bin_vals is None:
            print(binname, 'frequency is not 10 Hz')
            continue
        else:
            calibrated_values, timestamps, temp, freq = bin_vals
        
        #Output 1 min epoch compressed file
        print("Outputting 1 min epoch compressed file")
        compressed_table = epoch_compress(calibrated_values, timestamps, temp, freq, epoch=60)
        compressed_table.to_csv(outpath, sep='\t', index=False)
        
##        #Output timestamps
##        print("Outputting raw value file")
##        format_str = "%Y-%m-%d %H:%M:%S:%f"
##        startstamp = str(int(datetime.strptime(timestamps[0], format_str).timestamp()))
##        endstamp = str(int(datetime.strptime(timestamps[-1], format_str).timestamp()))
##        outstamps = startstamp + " " + endstamp
##        stamp_outpath =  baseroot + '/' + raw_table_root + '/' + outname[0:-4] + "_timestamps.txt"
##        open(stamp_outpath, 'w').write(outstamps)
##        
##        #Output rawxyz file
##        x_vals, y_vals, z_vals, light = calibrated_values
##        temp = expand_val(temp, 300)
##        
##        output_vals = [x_vals, y_vals, z_vals, temp, light]
##        names = ["X", "Y", "Z", "Temp", "Light"]
##        raw_xyz = pd.DataFrame(dict(zip(names, output_vals)))
##        
##        raw_table_outpath = baseroot + '/' + raw_table_root + '/' + outname
##        raw_xyz.to_csv(raw_table_outpath, index=False, sep='\t', columns = names)

