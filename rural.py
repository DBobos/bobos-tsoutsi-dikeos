# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:46:56 2021

@author: Παναγιώτης
"""

import os
from glob import glob
import pandas as pd
import numpy as np
import re
from functools import reduce

from sklearn.preprocessing import MinMaxScaler



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




# define the path to ntu simulation data
ntu_directory = os.path.join(cwd, 'Data', 'Simulator NTU')

# define the pattern for ntu rural-simulations data files
ntu_rural_pattern = re.compile('.*Rural.*LH.*\.txt')

# generate ntu rural simulator dataframe
df_ntu_rural = load_files(ntu_directory, ntu_rural_pattern)


# conversions
df_ntu_rural['Speed'] = (1000 / 3600) * df_ntu_rural['Speed'] # km/h -> m/s
df_ntu_rural['Time'] = df_ntu_rural['Time'] / 1000  # ms -> s
df_ntu_rural['HWay'] = df_ntu_rural['HWay'].replace(9999.9, np.nan)  # not to be aggregated
# add track label
df_ntu_rural['track'] = 'Motorway'
# select columns
df_ntu_rural = df_ntu_rural[['contestant_class',
                             'contestant',
                             'location',
                             'track',
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
                             'z-pos', 
                             'Time'
                             ]]
# rename columns
df_ntu_rural = df_ntu_rural.rename(columns={'Speed': 'speed_mps',
                                            'Acc': 'accelerator',
                                            'Gear': 'gear',
                                            'Brk': 'brake',
                                            'Wheel': 'steering',
                                            'HWay': 'headway_s',
                                            'ralpha': 'orientation',
                                            'x-pos': 'pos_x',
                                            'y-pos': 'pos_z',
                                            'z-pos': 'pos_y',
                                            'Time': 'time_s'
                                            }
                                   )



# calculate instant acceleration for df_ntu_rural
df_ntu_rural['acceleration'] = df_ntu_rural.groupby(['contestant'])['speed_mps'].diff() \
                               / df_ntu_rural.groupby(['contestant'])['time_s'].diff()

# calculate distance travelled by each contestant for df_ntu_rural
df_ntu_rural['distance'] = (df_ntu_rural['speed_mps'].shift() * df_ntu_rural.groupby(['contestant'])['time_s'].diff()) \
       + (df_ntu_rural['acceleration'] * (df_ntu_rural.groupby(['contestant'])['time_s'].diff() ** 2) / 2)

df_ntu_rural['acceleration'] = df_ntu_rural['acceleration'].replace(np.nan, 0)
df_ntu_rural['distance'] = df_ntu_rural['distance'].replace(np.nan, 0)


"""
ROUTE ESTIMATION AND DISPLACEMENT
"""

def distance_bins(df):
    """
    takes as input a dataframe and calculates the maximum cumulative distance
    travalled by any contestant and returns the dataframe with distance bins
    per 1 distance unit
    """
    # sort dataframe by contestant/distance and reset index 
    df = df.sort_values(['contestant', 'distance']).reset_index(drop=True)
    
    # calculate cumulative distance travelled per contestant
    df['cumulative_distance'] = df.groupby(['contestant'])['distance'].cumsum(axis=0)
    #dist = df.groupby((df.contestant != df.contestant.shift(1)).cumsum())['distance'].cumsum()
    #df['cumulative_distance'] = dist
    
    # create distance bins by 1 distance unit (m)
    bins = np.arange(0, df['cumulative_distance'].max(), 1)
    #df['bins'] = pd.cut(df['cumulative_distance'], bins, include_lowest=True)
    #df['bins'] = np.digitize(df['cumulative_distance'], bins)
    df['bins'] = np.searchsorted(bins, df['cumulative_distance'].values)

    return df


def route_estimation(df):
    """
    takes as input a dataframe with the entries of all contestants on a specific
    track and simulator and returns an approximation of the route of the track
    which is calculated by the mean coordinates of the control group within
    each distance bin
    """
    # filter control group entries
    df = df[df['contestant_class'] == 'Control']
    
    # calculate cumulative distance and distance bins
    df = distance_bins(df)
    
    # calculate mean X and Y values per distance bin
    df['route_x'] = df.groupby(['bins'])['pos_x'].transform('mean')
    df['route_y'] = df.groupby(['bins'])['pos_y'].transform('mean')
    df = df [['bins', 'route_x', 'route_y']]
    
    # drop duplicate rows
    df = df.drop_duplicates(inplace=False)
    
    return df


def positioning(df):
    """
    takes as input a dataframe with the entries of all contestants on a specific
    track and simulator and returns the mean values of coordinates for each
    contestant in each distance bin
    """
    
    # calculate cumulative distance and distance bins
    df = distance_bins(df)

    # calculate mean X and Y values per contestant and distance bin
    df['mean_x'] = df.groupby(['bins', 'contestant'])['pos_x'].transform('mean')
    df['mean_y'] = df.groupby(['bins', 'contestant'])['pos_y'].transform('mean')
    
    return df


def lane_off(df):
    
    df.loc[(df.groupby('contestant')['mean_x'] != df.groupby('contestant')['mean_x'].shift()) \
           & (df.groupby('contestant')['mean_y'] != df.groupby('contestant')['mean_y'].shift()), 'lane_offset_m'] = \
    np.sqrt((df['mean_x'] - df['route_x'])**2 + (df['mean_y'] - df['route_y'])**2) 


    df.loc[(df.groupby('contestant')['mean_x'] != df.groupby('contestant')['mean_x'].shift()) \
           & (df.groupby('contestant')['mean_y'] == df.groupby('contestant')['mean_y'].shift()), 'lane_offset_m'] = \
    abs(df['mean_x'] - df['route_x']) 

    df.loc[(df.groupby('contestant')['mean_x'] == df.groupby('contestant')['mean_x'].shift()) \
           & (df.groupby('contestant')['mean_y'] != df.groupby('contestant')['mean_y'].shift()), 'lane_offset_m'] = \
    abs(df['mean_y'] - df['route_y'])
    
    df = df.fillna(method='ffill') 
    
    return df



# calculate route coordinates for df_ntu_rural
route_ntu_rural = route_estimation(df_ntu_rural)

# calculate each contestant's positioning for df_ntu_rural
df_ntu_rural = positioning(df_ntu_rural)

# get route coordinates per distance bin on df_ntu_rural
df_ntu_rural = pd.merge(df_ntu_rural, route_ntu_rural, on='bins', how='inner')

# calculate the lane offset in meters as the distance of each contestant's
# position from the route coordinates, for df_ntu_rural
df_ntu_rural = lane_off(df_ntu_rural)



# normalize brake to 0-1
df_ntu_rural['brake'] = MinMaxScaler().fit_transform(np.array(df_ntu_rural['brake']).reshape(-1,1))
# normalize headway to 0-1
df_ntu_rural['headway_s'] = MinMaxScaler().fit_transform(np.array(df_ntu_rural['headway_s']).reshape(-1,1))
# normalize steering to -1 - 1
df_ntu_rural['steering'] = df_ntu_rural['steering'].abs()
df_ntu_rural['steering']= df_ntu_rural['steering'] /df_ntu_rural['steering'].abs().max()
# lane_offset normalize to -1 - 1
df_ntu_rural['lane_offset_m']= df_ntu_rural['lane_offset_m'] /df_ntu_rural['lane_offset_m'].abs().max()

# export df to csv
df_ntu_rural.to_csv('df_ntu_rural.csv', index=True)



# define the list of the desired columns for the rural dataframe
rural_columns = ['contestant_class',
                 'contestant',
                 'location',
                 'track',
                 'event_class',
                 'time_s',
                 'distance',
                 'cumulative_distance',
                 'speed_mps',
                 'acceleration',
                 'accelerator',
                 'gear',
                 'brake',
                 'steering',
                 'headway_s',
                 'lane_offset_m',
                 'orientation'
                 ]






# define the list of datasets that will be used to generate the final dataframe
# for the 'rural' scenario
rural_dataframes = [
                    df_ntu_rural[[c for c in rural_columns]]
                    ]

# create the final dataframe (df_u) for the 'rural' scenario
df_u = pd.concat(rural_dataframes, ignore_index=True)

# calculate mean values for the 'rural' scenario
df_u_mean = df_u.groupby(['contestant_class',
                          'contestant',
                          'location'
                          ], as_index=True
                         )[['speed_mps',
                            'acceleration',
                            'accelerator',
                            'gear',
                            'brake',
                            'steering',
                            'headway_s',
                            'lane_offset_m',
                            'orientation']].mean()

# calculate standard deviation for the 'rural' scenario
df_u_std = df_u.groupby(['contestant_class',
                          'contestant',
                          'location'
                          ], as_index=True
                         )[['speed_mps',
                            'acceleration',
                            'accelerator',
                            'gear',
                            'brake',
                            'steering',
                            'headway_s',
                            'lane_offset_m',
                            'orientation']].std()

                            

def rename_columns(df, prefix, suffix):
    """
    takes as inputs a dataframe, a prefix and a suffix and renames the columns
    of the dataframe adding these values before and after the column name
    respectively
    """
    new_col_names = [(i, prefix + '_' + i + '_' + suffix) for i in df.columns.values]
    df.rename(columns = dict(new_col_names), inplace=True)
    
    return df


# rename the columns to join the datasets
df_u_mean = rename_columns(df_u_mean, 'rural', 'mean')
df_u_std = rename_columns(df_u_std, 'rural', 'std')

# create a list of above dataframes in order to merge them
dataframes = [df_u_mean, df_u_std]

# merge dataframes into one
df_agg = reduce(lambda  left, right: pd.merge(left, right,
                                              on=['contestant_class', 'contestant', 'location' ],
                                              how='outer'), dataframes)

# sort df_agg columns by name
df_agg = df_agg.reindex(sorted(df_agg.columns), axis=1)

# export df_agg to csv
df_agg.to_csv('aggregated_data_rural.csv', index=True)
