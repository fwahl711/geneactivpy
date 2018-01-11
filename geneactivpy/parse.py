import logging
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd
import re


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
