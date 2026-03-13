# import libraries
import os
from glob import glob
import pandas as pd
import numpy as np
import re


# get path of current working directory
# ! it is important to set current working directory to the folder
# ! where the script and data folder are located
cwd = os.getcwd()


def file_info(file):
    """
    takes as input a simulator data file and returns a dictionary with the
    filename, the class (Control/Patient), the location (HMU/NTU) and the
    contestant id
    """
    file_classified = dict()
    file_classified['file'] = file
    
    # classify files between control group and patients
    if re.match(r'.*Simulator\s((?:NTU)|(?:HMU))\\Controls\\.*', file):
        file_classified['contestant_class'] = 'Control'
    elif re.match(r'.*Simulator\s((?:NTU)|(?:HMU))\\Patients\\.*', file):
        file_classified['contestant_class'] = 'Patient'
    
    # get the location (HMU/NTU)
    if re.match(r'.*Simulator\s(NTU).*', file):
        file_classified['location'] = 'NTU'
    elif re.match(r'.*Simulator\s(HMU).*', file):
        file_classified['location'] = 'HMU'
    
    # split filename to get the contestant id
    filename_split = file_classified['file'].split('\\')

    # get the contestant id
    file_classified['contestant'] = re.split(r'[-,\s,_]', filename_split[-1])[0]
    
    return file_classified


def load_xls(filepath):
    """
    takes a path to an xlsx file, reads it and returns a dataframe with the
    values classified based on the time they occured in relation to the events
    """
    # read xlsx data file
    df = pd.read_excel(filepath)
    
    # locate the first empty row
    first_empty_row = df[df.isnull().all(axis=1) == True].index.tolist()[0]
    
    # remove everything from this line and below
    df = df.loc[0:first_empty_row-1]
    
    # drop the columns where all elements are NaN
    df = df.dropna(axis=1, how='all')
    
    # define event_class as 'Before events' as there are no events in HMU data
    df['event_class'] = 'Before events'
    
    # get info about the processed data file
    file_classified = file_info(filepath)
    df['location'] = file_classified['location']
    df['contestant'] = file_classified['contestant']
    df['contestant_class'] = file_classified['contestant_class']
    
    return df


def load_txt(filepath):
    """
    takes a path to a txt file, reads it and returns a dataframe with the
    values classified based on the time they occured in relation to the events
    """
    # read txt data file
    df = pd.read_csv(filepath,
                       delimiter = '\t',
                       usecols=[*range(0,28)],
                       skiprows=2
                       )
    
    # locate the row where the first event occurs or set this variable to the
    # last line if there are no events
    first_event_row = (df[df['EvDist'] < 99999].index.tolist() or [df.index[-1]])[0]

    # create a list of conditions to define the class of each row based on the
    # timing in relation to events
    conditions = [
        (df['EvDist'] >= 99999) & (df.index <= first_event_row),
        (df['EvDist'] >= 99999) & (df.index > first_event_row),
        (df['EvDist'] < 99999)
        ]

    # create a list of the values to assign on dataframe rows for each condition
    values = ['Before events', 'Inbetween events', 'During events']
    
    # define event_class and assign values to it using the above lists as arguments
    df['event_class'] = np.select(conditions, values)
    
    # get info about the processed data file
    file_classified = file_info(filepath)
    df['location'] = file_classified['location']
    df['contestant'] = file_classified['contestant']
    df['contestant_class'] = file_classified['contestant_class']
    
    return df


def load_files(directory, pattern):
    """
    takes as input a directory and a pattern and returns a dataframe with the
    column values of all the files in the directory and its
    subdirectories whose filename matches the pattern
    """
    # get a list of all subdirectories
    subdirs = os.listdir(directory)
    
    # create a list to store all files in the subdirectories
    files = []
    
    # find all files in the subdirectories
    for subdir in subdirs:
        subdirpath = os.path.join(directory, subdir)
        for dir,_,_ in os.walk(subdirpath):
            files.extend(glob(os.path.join(dir, '*.*')))
    
    # filter files that match the defined pattern
    files = list(filter(pattern.match, files))

    # iterate through files and generate a dataframe with all entries
    temp_df = pd.DataFrame()
    df = pd.DataFrame()
    for file in files:
        if (file.endswith('.xlsx')) or (file.endswith('.xls')):
            temp_df = load_xls(file)
        elif file.endswith('.txt'):
            temp_df = load_txt(file)
        # save the column names from the first dataset, assuming that for a
        # simulator and track, the columns will match 
        if files.index(file) == 0:
            df_headers = temp_df.columns.values.tolist()
        # replace headers on temp_df with the column index in order to avoid
        # conflicts while appending
        temp_df = temp_df.rename(columns=lambda c: temp_df.columns.get_loc(c))
        
        # append each temp_df to the final df
        df = df.append(temp_df, ignore_index=True)

    # set the column names of the final df
    df.columns = df_headers
    
    return df


# define the path to hmu simulation data
hmu_directory = os.path.join(cwd, 'Data', 'Simulator HMU')

# define the path to ntu simulation data
ntu_directory = os.path.join(cwd, 'Data', 'Simulator NTU')

# define the pattern for hmu motorway-simulations data files
hmu_mtway_pattern = re.compile('.*Motorway.*\.xls.')

# define the pattern for hmu urban-low-traffic-simulation data files
hmu_urblt_pattern = re.compile('.*Urban-L.*\.xls.')

# define the pattern for ntu motorway-simulations data files
ntu_mtway_pattern = re.compile('.*Motorway.*\.txt')

# define the pattern for hmu urban-low-traffic-simulation data files
ntu_urblt_pattern = re.compile('.*Urban.*LH.*\.txt')

# generate hmu motorway simulator dataframe
df_hmu_mtway = load_files(hmu_directory, hmu_mtway_pattern)
# select columns
df_hmu_mtway = df_hmu_mtway[['contestant_class',
                             'contestant',
                             'location',
                             'event_class',
                             'speed (km/h)',
                             'accelerator',
                             'gear',
                             'brake',
                             'steering',
                             'foll_dist_in_seconds',
                             'lane_offset (m)'
]]
# rename columns
df_hmu_mtway = df_hmu_mtway.rename(columns={'speed (km/h)': 'speed_kmph',
                                            'foll_dist_in_seconds': 'headway_s',
                                            'lane_offset (m)': 'lane_offset_m'})

# generate hmu urban-low-traffic simulator dataframe
df_hmu_urblt = load_files(hmu_directory, hmu_urblt_pattern)
# select columns
df_hmu_urblt = df_hmu_urblt[['contestant_class',
                             'contestant',
                             'location',
                             'event_class',
                             'current_speed',
                             'accel',
                             'current_gear',
                             'brake_pedal_posn',
                             'steering_posn',
                             'current_time_headway',
                             'orientation',
                             'pos_x',
                             'pos_y',
                             'pos_z'
]]
# rename columns
df_hmu_urblt = df_hmu_urblt.rename(columns={'current_speed': 'speed_kmph',
                                            'accel': 'accelerator',
                                            'current_gear': 'gear',
                                            'brake_pedal_posn': 'brake',
                                            'steering_posn': 'steering',
                                            'current_time_headway': 'headway_s'
                                            })

# generate ntu motorway simulator dataframe
df_ntu_mtway = load_files(ntu_directory, ntu_mtway_pattern)
# select columns
df_ntu_mtway = df_ntu_mtway[['contestant_class',
                             'contestant',
                             'location',
                             'event_class',
                             'Speed',
                             'Acc',
                             'Gear',
                             'Brk',
                             'Wheel',
                             'HWay',
                             'ralpha',
                             'x-pos',
                             'y-pos',
                             'z-pos'
]]
# rename columns
df_ntu_mtway = df_ntu_mtway.rename(columns={'Speed': 'speed_kmph',
                                            'Acc': 'Accelerator',
                                            'Gear': 'gear',
                                            'Brk': 'brake',
                                            'Wheel': 'steering',
                                            'HWay': 'headway_s',
                                            'ralpha': 'orientation',
                                            'x-pos': 'pos_x',
                                            'y-pos': 'pos_y',
                                            'z-pos': 'pos_z'
                                            })

# generate ntu urban-low-traffic simulator dataframe
df_ntu_urblt = load_files(ntu_directory, ntu_urblt_pattern)
df_ntu_urblt = df_ntu_urblt[['contestant_class',
                             'contestant',
                             'location',
                             'event_class',
                             'Speed',
                             'Acc',
                             'Gear',
                             'Brk',
                             'Wheel',
                             'HWay',
                             'ralpha',
                             'x-pos',
                             'y-pos',
                             'z-pos'
]]
# rename columns
df_ntu_urblt = df_ntu_urblt.rename(columns={'Speed': 'speed_kmph',
                                            'Acc': 'Accelerator',
                                            'Gear': 'gear',
                                            'Brk': 'brake',
                                            'Wheel': 'steering',
                                            'HWay': 'headway_s',
                                            'ralpha': 'orientation',
                                            'x-pos': 'pos_x',
                                            'y-pos': 'pos_y',
                                            'z-pos': 'pos_z'
                                            })

