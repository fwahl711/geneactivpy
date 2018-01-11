import logging
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd
import os
import re

class Patient:
    def __init__(self,path_binary=None,path_processed=None,last_step=None,write_intermediary=False):
        self.path_binary=path_binary
        self.path_processed=path_processed
        if self.path_binary==None and self.path_processed==None:
            logging.error("A file is required to create a class instance. Specify either 'path_binary' or 'path_processed'.")
            return

        # Last step do to
        self.termination_step=last_step
        # Whether to create intermediary files if True
        ## else save after the last step
        self.write_intermediary=write_intermediary
        if self.path_processed:
            # Read processed file into Dataframe
            self.type_processed=self.get_type_file()
        elif self.path_binary:
            self.process_binary()

        self.calibration=None
        self.basic_info=None
        self.df=None

    def __add__(self,other):
        """
        Overload the '+' to merge the files
        """
        pass

    def process_binary(self):
        """
        Processes the binary file.
        Stops at the step that is given at the start.
        """
        # Open file
        with open(path_binary_file,'r') as f:
            data = [line.strip() for line in f]

        # Extract the basic device information
        self.get_basic_device_info(data)
        if self.termination_step=='basic':
            return
        # Extract raw values
        self.get_raw_values(data)
        if self.termination_step=='raw':
            return
        # Calibrate values
        self.calibrate_df()
        if self.termination_step=='calibrate':
            return
        # Roll values (smooths out values)
        self.roll_window(operation='median',shift=True,window_seconds=5)
        if self.termination_step=='roll':
            return
        # Calculate wrist angles
        self.calc_angle()

        # Get sleep score

        # Compress dataframe
        pass

    def get_type_file(self):
        file_path=self.path_processed
        basename=os.path.basename(file_path) 
        ext=basename.split(".")[-1]
        return ext
        
    def get_basic_device_info(self,data):
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
        self.basic_info= calibration_dict

    def calc_block_start_and_step(self,ref_string,recording_data):
        """
        Calculate the step between each block (page) of recorded data as well as in the index for the start of the recording blocks.
        """
        index1=recording_data.index(ref_string)
        index2=recording_data.index(ref_string,index1+1)
        return index1, index2-index1

    def parse_one_block(self,recording_block_list):
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
        df= self.hex_to_int_df(hexstring,page_time)
    
        # For all these points in time, we consider the temperature to be the same (add temp column)
        df["temperature"]=temperature
    
        return df
    
    def hex_to_int_df(self,hexstream,start_timestamp):
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

    def get_raw_values(self,data):
        """
        Find all blocks of recorded data in the binary file, send them to be parsed and combine the values from each block afterwards.
        Multiprocessing elegible.
        """

        # Get information required to split data in blocks
        ## Get index of the start of the blocks 
        block_start, step = self.calc_block_start_and_step("Recorded Data", data)
        ## List containing multiple pairs of block + calibration data
        tmp_data=data[block_start:]
        blocks_data=[tmp_data[i:i+step] for i in range(0,len(tmp_data),step)]

        with Pool(multiprocessing.cpu_count()-1) as p:
            r=p.map_async(self.parse_one_block,blocks_data)
            r.wait()
        returned_dfs=r.get()
   
        # Concatenate all the small dfs from the individual block parsing
        self.df=pd.concat(returned_dfs)

    def validate_columns(self,required_columns=["x","y","z","temperature"]):
        columns=self.df.columns
        for elem in required_columns:
            if elem not in columns:
                return False
        return True

    def calibrate_df(self):
        """
        Takes the dataframe with the x,y,z,light and temperature column.
        Using the calibration dict, it adjusts the xyz columns
        """
        if not self.validate_columns(required_columns=['x','y','z','light']):
            logging.error("A column is missing in the dataframe for the calibration to occur.")
            return
        
        self.df["x"]=(self.df["x"]*100-self.basic_info["x_offset"])/self.basic_info["x_gain"]
        self.df["y"]=(self.df["y"]*100-self.basic_info["y_offset"])/self.basic_info["y_gain"]
        self.df["z"]=(self.df["z"]*100-self.basic_info["z_offset"])/self.basic_info["z_gain"]
        self.df["light"]=self.df["light"]*self.basic_info["lux"]/self.basic_info["volts"]
        self.calibration=True
        

    def find_datetime_shift(self,seconds=5):
        """
        Rolling is done from the right, if you want from the left
        it requires to find the number of elements to shift by.
        """
        shift_val=1
        delta_t=timedelta(seconds=seconds)
        while True:
            if (self.df.index[shift_val]-self.df.index[0])>=delta_t:
                return shift_val-1
            elif shift_val>10**4:
                break
            else:
                shift_val+=1
    
    
    def roll_window(self,operation='median',shift=True,window_seconds=5):
        """
        Takes a dataframe and creates rolling windows. Using those
        windows, it computes either the median, mean or std of each window.
        """
        if window_seconds>0:
            roll_seconds="{}S".format(window_seconds)
        else:
            return None
        
        # If not dataframe
        if not isinstance(self.df,pd.DataFrame):
            return None
            
        rolled=self.df.rolling(roll_seconds)
        
        # Apply operation
        if operation=='median':
            self.df=rolled.median()
        elif operation=='mean':
            self.df=rolled.mean()
        elif operation=='std':
            self.df=rolled.std()
        else:
            return None
        
        # If you want the windows to be calculated on the left (rather than the right)
        if shift:
            shift_val=-1*self.find_datetime_shift(seconds=window_seconds)
            self.df=self.df.shift(shift_val)

    def calc_angle(self):
        """
        Calculate the angle given the x, y and z values.
        Equation:
            atan(z/sqrt(x^2 +y^2)) *180/pi
        """
        if not self.validate_columns(required_columns=['x','y','z']):
            logging.error("Cannot calculate the angle, a column is missing.")
            return
        
        angle=self.df['z']/np.sqrt(np.power(self.df['x'],2)+np.power(self.df['y'],2))
        angle=np.arctan(angle)*180/np.pi
        return angle
