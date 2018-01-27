import click
import logging
from multiprocessing import Pool
from datetime import timedelta
import multiprocessing
import numpy as np
import pandas as pd
import os
import re

"""
Class that does the bulk of the work.
"""

class Patient:
    def __init__(self,path_binary=None,path_processed=None,endpoint="compress",write_intermediary=False,compress_minutes=5,patient_name=None):
        self.calibration=None
        self.basic_info=None
        self.df=None
        self.angles=None
        self.inactivity=None
        self.inactivity_compressed=False
        self.dev_sleep=None

        self.latest_df=None
        self.compress_minutes=compress_minutes

        self.path_binary=path_binary
        self.path_processed=path_processed
        if self.path_binary is None and self.path_processed is None:
            logging.warning("A file was not given in creation of a class instance. Specify either 'path_binary' or 'path_processed'.")
            return

        # set filename base (take patient_name)
        self.fn=patient_name
        if not self.fn:
            if self.path_processed:
                self.fn="".join([str(i) for i in os.path.basename(self.path_processed).split(".")[:-1]])
            elif self.path_binary:
                self.fn="".join([str(i) for i in os.path.basename(self.path_binary).split(".")[:-1]])

        # Last step do to
        self.endpoint=endpoint
        # Whether to create intermediary files if True
        ## else save after the last step
        self.write_intermediary=write_intermediary

        if self.path_processed:
            # Read processed file into Dataframe
            self.type_processed=self.get_type_file()
        elif self.path_binary:
            self.process_binary()


    def __add__(self,other):
        """
        Overload the '+' to merge the dataframes.
        """
        if not isinstance(other,Patient):
            logging.warning("Attempting to add a Patient with a {}.".format(type(other)))
            return self.latest_df
        if other.latest_df is None:
            logging.info("The second Patient object does not have a value set for latest_df.")
            return self.latest_df
        if not (isinstance(self.latest_df,pd.DatetimeIndex) and isinstance(other.latest_df,pd.DatetimeIndex)):
            logging.warning("`merge_tables`: one of the data structure passed does not have an datetime as the index.")
            return None

        ## Determine which file comes first
        if self.latest_df.index[0]>other.latest_df.index[0]:
            first=other.latest_df
            second=self.latest_df
        elif df1.index[0]<df2.index[0]:
            first=self.latest_df
            second=other.latest_df
        else:
            logging.warning("`__add__`: both dataframes start at the same timepoint.")
            if len(first)>len(second):
                return first
            else:
                return second

        ## Determine if there is overlap in the beginning and end of the recordings
        if first.index[-1]>second.index[0]:
            # Overlap, apply function that determines action (with option to change default behaviour)
            first,second=self.overlap_action(first,second,behaviour="half")

        if first and second:
            appended_df=pd.concat([first,second],axis=0,join='outer',ignore_index=False)
            return appended_df
        else:
            return first

    def overlap_correction(self,first,second,behaviour='half'):
        """
        Deal with overlapping data.
        Need to find intersect (section of interest).
        Apply given behaviour, where to cut.
        """
        # Get index of intersect
        intersect_end_first=first.index.intersection(second.index)
        intersect_begin_second=second.index.intersection(first.index)
        len_intersect=len(intersect_end_first)

        # Remove equally in both data structures
        if behaviour=='half':
            n_elem_cut=len_intersect//2
            first=first.drop(index=intersect_end_first[n_elem_cut:],axis=0)
            second=second.drop(index=intersect_begin_second[:n_elem_cut],axis=0)
        elif behaviour=='first':
            first=first.loc[first.index[0]:intersect_end_first[0]]
        elif behaviour=='second':
            second=second.loc[intersect_begin_second[-1]:second.index[-1]]
        else:
            logging.warning("`overlap_correction`: invalid behaviour on how to deal with overlap.")
            return None,None

        # Start looking from the end of the intersect_end_first
        # and from the beginning of the intersect_begin_second
        return first,second

    def process_binary(self):
        """
        Processes the binary file.
        Stops at the step that is given at the start.
        Ordered steps:
            - Get basic info
            - Get raw values (x,y,z,light,temp)
            - Calibrate values (x,y,z,light)
            - Use rolling window (operation median) to smooth data
            - Calculate angle with smoothed data
            - Determine periods of inactivity by looking at windows of time where
        """
        # Open file
        with open(self.path_binary,'r') as f:
            data = [line.strip() for line in f]

        # Extract the basic device information
        self.get_basic_device_info(data)
        if self.endpoint=='basic':
            return
        # Extract raw values
        self.get_raw_values(data)
        if self.endpoint=='raw':
            return
        # Calibrate values
        self.calibrate_df()
        if self.endpoint=='calibrate':
            return
        # Roll values (smooths out values)
        self.roll_window(operation='median',shift=True,window_seconds=5)
        if self.endpoint=='roll':
            return
        # Calculate wrist angles
        ## and average angles over 5 seconds
        self.calc_angle()
        self.angles=self.compress_windows_df(self.angles,c_w_seconds=5,operation='mean')
        if self.endpoint=='angles':
            return
        # Get sleep score
        self.determine_activity()
        if self.endpoint=='inactivity':
            return

        # Compress dataframe
        self.inactivity=self.compress_windows_df(self.inactivity,c_w_minutes=self.compress_minutes,operation='sum')
        self.inactivity_compressed=True
        self.latest_df=self.inactivity
        if self.endpoint=='compress':
            return

    def get_type_file(self):
        file_path=self.path_processed
        basename=os.path.basename(file_path)
        ext=basename.split(".")[-1]
        return ext

    def get_basic_device_info(self,data):
        """
        Takes in list of lines from binary file and returns a dictionary containing calibration information (gain,offset,volts,lux) and frequency of device
        """
        click.echo("\t Getting basic info.")

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
                if page_time is None:
                    index_start_t=tmp.index(":")+1
                    time_str=tmp[index_start_t:].strip()
                    page_time=pd.to_datetime(time_str,format="%Y-%m-%d %H:%M:%S:%f")

            elif tmp.startswith("Temperature"):
                if temperature is None:
                    temperature=float(tmp.split(":")[1])

            elif tmp.startswith("Measurement Frequency"):
                if hexstring is None:
                    # Add one to the index, the hexstring is the one after Measurement Frequency
                    hexstring=recording_block_list[index+1].strip()
                    break

        # If it can't find either of the variables
        if temperature is None or page_time is None or hexstring is None:
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
        Multiprocessing enabled.
        """
        click.echo("\t Getting raw values.")

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
        self.latest_df=self.df

    def validate_columns(self,dataframe=None,required_columns=["x","y","z","temperature"]):
        if dataframe is None:
            dataframe=self.df
        columns=dataframe.columns
        for elem in required_columns:
            if elem not in columns:
                return False
        return True

    def calibrate_df(self):
        """
        Takes the dataframe with the x,y,z,light and temperature column.
        Using the calibration dict, it adjusts the xyz columns
        """
        click.echo("\t Calibrating values.")
        if not self.validate_columns(required_columns=['x','y','z','light']):
            logging.error("A column is missing in the dataframe for the calibration to occur.")
            return

        self.df["x"]=(self.df["x"]*100-self.basic_info["x_offset"])/self.basic_info["x_gain"]
        self.df["y"]=(self.df["y"]*100-self.basic_info["y_offset"])/self.basic_info["y_gain"]
        self.df["z"]=(self.df["z"]*100-self.basic_info["z_offset"])/self.basic_info["z_gain"]
        self.df["light"]=self.df["light"]*self.basic_info["lux"]/self.basic_info["volts"]
        self.calibration=True
        self.latest_df=self.df

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
        click.echo("\t Rolling df on a window of {} secs and applying `{}`".format(window_seconds,operation))
        if window_seconds>0:
            roll_seconds="{}S".format(window_seconds)
            logging.info("`roll_window`: roll_seconds={}".format(roll_seconds))
        else:
            return None

        # If not dataframe
        if not isinstance(self.df,pd.DataFrame):
            logging.warning("`df` is not a Pandas DataFrame instance.")
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
        self.latest_df=self.df

    def calc_angle(self):
        """
        Calculate the angle given the x, y and z values.
        Equation:
            atan(z/sqrt(x^2 +y^2)) *180/pi
        """
        click.echo("\t Calculating wrist angles.")
        # Check if self.df has been calculated
        if self.latest_df is None:
            logging.warning("Dataframe `df` is not defined. Probably skipped a step.")
            return

        if not self.validate_columns(required_columns=['x','y','z']):
            logging.error("Cannot calculate the angle, a column is missing.")
            return

        self.angles=self.latest_df['z']/np.sqrt(np.power(self.latest_df['x'],2)+np.power(self.latest_df['y'],2))
        self.angles=np.arctan(self.angles)*180/np.pi
        self.latest_df=self.angles

    def determine_activity(self,window_minutes=5,diff_angle_threshold=5):
        """
        Computes the change of angles between each point (avg of 5seconds)
        and then rolls on windows of size window_minutes to determine if
        the max change is bigger than angles_threshold or if the -1 * min
        change is bigger than angles_threshold. Inactivity is when the
        values are between -threshold and +threshold.
        """
        click.echo("\t Determining inactivity with a rolling\n\t window of {} mins and with\n\t a threshold of {} degrees.".format(window_minutes,diff_angle_threshold))
        try:
            # Checking as to the type of input
            if not isinstance(self.angles,pd.Series):
                raise ValueError
        except NameError:
            logging.error("Angles has not been calculated.")
            return
        except ValueError:
            logging.error("Angles is not a pandas series. An error has occured somewhere. Find it.")
            return

        # Rolling rule prep
        if (isinstance(window_minutes, int) or isinstance(window_minutes,float)) and window_minutes>0:
            roll_rule="{:.2g}T".format(window_minutes)
            logging.info("'determine_activity': window for the roll=  {} minutes".format(roll_rule))
        else:
            loggin.error("'determine_activity': window_minutes is not a valid input [{}]".format(window_minutes))
            return None

        # Absolute value of change between angle
        diff_angles=self.angles.diff().abs()

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

        self.inactivity=diff_angles.rolling(roll_rule).apply(window_inactivity)
        self.latest_df=self.inactivity

    def compress_windows_df(self,data,c_w_minutes=0,c_w_seconds=0,operation='sum'):
        """
        Takes a dataframe/serie, groups tries to group by `c_w_minutes` if not
        with `c_w_seconds` and else it fails.
        Then after resampling the data, it applies `operation` on it.
        `operation`: sum, mean, std, median, count
        """
        click.echo("\t Compressing df and applying a `{}`".format(operation))
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

    def compute_dev_sleep(self,c_w_minutes=1):
        """
        Sets up dataframe with the time + x_dev + y_dev + z_dev + temperature
            + light + sleep score
        """

        # Calculate x_dev, y_dev and z_dev over c_w_minutes window
        self.dev_sleep=self.compress_windows_df(self.df[['x','y','z']],c_w_minutes=c_w_minutes,operation="std")
        self.dev_sleep=self.dev_sleep.rename(index=str,columns={"x":"x_dev","y":"y_dev","z":"z_dev"})

        self.dev_sleep['time']=self.dev_sleep.index
        self.dev_sleep[['temp','light']]=self.compress_windows_df(self.df[['temperature','light']],c_w_minutes=c_w_minutes,operation='mean')
        self.dev_sleep['sleep score']=self.compress_windows_df(self.inactivity,c_w_minutes=c_w_minutes,operation='mean')

    def write_inactivity(self,output_directory,patient_code=None,time_format="%Y-%m-%d %H:%M:%S.%f"):
        """
        Writes the inactivity dataframe to the directory.
        """

        if patient_code is None:
            patient_code=self.fn
        try:
            if self.inactivity is None:
                raise NameError
        except NameError:
            click.echo("Attempting to write a variable that does not exists.")
            logging.error("Attempting to write a variable that does not exists. ")
            return None
        if not (isinstance(self.inactivity,pd.DataFrame) or isinstance(self.inactivity,pd.Series)):
            logging.error("The input is not an instance of a DataFrame or a Series [type={}]".format(str(type(df))))
            return None

        # If patient_code folder does not exist create it
        patient_code=patient_code.strip().replace(" ","_")
        patient_path=os.path.join(output_directory,patient_code)

        # Create directory if it does not exist
        if not os.path.exists(patient_path):
            os.mkdir(patient_path)
        file_name=patient_code+"___inactivity.csv"
        file_path=os.path.join(patient_path,file_name)
        logging.info("`to_csv` is writing to {}.".format(file_path))
        click.echo("Writing file out to {}".format(file_path))

        self.inactivity.to_csv(file_path,index=True,date_format=time_format,header=True)

    def write_dev_sleep(self,output_directory,patient_code=None,time_format="%Y-%m-%d %H:%M:%S"):
        """
        Writes csv with the time + x,y,z standard deviations +
            temperature + light + sleep scores
        The values are averages for 1 minute.
        """

        if self.dev_sleep is None:
            self.compute_dev_sleep()

        if patient_code is None:
            patient_code=self.fn
        try:
            if self.dev_sleep is None:
                raise NameError
        except NameError:
            click.echo("Attempting to write a variable that does not exists.")
            logging.error("Attempting to write a variable that does not exists. ")
            return None

        # If patient_code folder does not exist create it
        patient_code=patient_code.strip().replace(" ","_")
        patient_path=os.path.join(output_directory,patient_code)

        # Create directory if it does not exist
        if not os.path.exists(patient_path):
            os.mkdir(patient_path)
        file_name=patient_code+"___dev_sleep.csv"
        file_path=os.path.join(patient_path,file_name)
        logging.info("`to_csv` is writing to {}.".format(file_path))
        click.echo("Writing file out to {}".format(file_path))

        #
        self.dev_sleep.to_csv(file_path,index=False,date_format=time_format,header=True)
