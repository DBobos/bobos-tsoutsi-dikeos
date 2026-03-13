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
        file_classified['class'] = 'Control'
    elif re.match(r'.*Simulator\s((?:NTU)|(?:HMU))\\Patients\\.*', file):
        file_classified['class'] = 'Patient'
    
    # get the location (HMU/NTU)
    if re.match(r'.*Simulator\s(NTU).*', file):
        file_classified['location'] = 'NTU'
    elif re.match(r'.*Simulator\s((?:NTU)|(?:HMU))\\Patients\\.*', file):
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
    
    # classify records based on their timing in correlation to events
    '''
    ---> PENDING CLARIFICATION ON HOW EVENTS ARE DEFINED IN HMU FILES <---
    '''
    
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
    
    # define EventClass and assign values to it using the above lists as arguments
    df['EventClass'] = np.select(conditions, values)
    
    file_classified = file_info(filepath)
    
    df['location'] = file_classified['location']
    df['contestant'] = file_classified['contestant']
    df['class'] = file_classified['class']
    
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
    df = pd.DataFrame()
    for file in files:
        if (file.endswith('.xlsx')) or (file.endswith('.xls')):
            df = df.append(load_xls(file), ignore_index=True)
        elif file.endswith('.txt'):
            df = df.append(load_txt(file), ignore_index=True)
            
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

"""
PENDING LINE 64
# generate hmu motorway simulator dataframe
df_hmu_mtway = load_files(hmu_directory, hmu_mtway_pattern)

# generate hmu urban-low-traffic simulator dataframe
df_hmu_urblt = load_files(hmu_directory, hmu_urblt_pattern)
"""

# generate ntu motorway simulator dataframe
df_ntu_mtway = load_files(ntu_directory, ntu_mtway_pattern)

# generate ntu urban-low-traffic simulator dataframe
df_ntu_urblt = load_files(ntu_directory, ntu_urblt_pattern)

# generate ntu motorway simulator dataframe with average values
df_ntu_mtway_avg = df_ntu_mtway.groupby(['contestant', 'EventClass']).mean()

# generate ntu urban-low-traffic simulator dataframe with average values
df_ntu_urblt_avg = df_ntu_urblt.groupby(['contestant', 'EventClass']).mean()

"""
Finding displacement
"""

# function to compute the distance every 30 millisecs
def lateral_pos(df):
    df['Distance'] = df['Speed'] * 0.03
    return df

#apply above function to ntu_mtway 
df_ntu_mtway = lateral_pos(df_ntu_mtway)

#compute the cumulative distance for each contestant
a = df_ntu_mtway.groupby((df_ntu_mtway.contestant != df_ntu_mtway.contestant.shift(1)).cumsum())['Distance'].cumsum()

#create column in original df for cumulative distance
df_ntu_mtway['cum_sum'] = a

#create 100m bins
bins = np.arange(0,df_ntu_mtway['cum_sum'].max(), 100)

#create column in original df for bins based on cumulative distance
df_ntu_mtway['bins'] = pd.cut(df_ntu_mtway['cum_sum'], bins, include_lowest=True)

#compute mean x, mean y for each contestant for every bin 
df_ntu_mtway['meansx'] = df_ntu_mtway.groupby(['bins', 'contestant'])['x-pos'].transform('mean')
df_ntu_mtway['meansy'] = df_ntu_mtway.groupby(['bins', 'contestant'])['y-pos'].transform('mean')

######### ROUTE ESTIMATION #########

# create a df with control group from original df
mean_control = df_ntu_mtway[df_ntu_mtway['class'] == 'Control']

#apply above function to mean_control 
mean_control = lateral_pos(mean_control)

#compute the cumulative distance for each contestant
a = mean_control.groupby((mean_control.contestant != mean_control.contestant.shift(1)).cumsum())['Distance'].cumsum()

#create column in original df for cumulative distance
mean_control['cum_sum'] = a

#create 100m bins
bins = np.arange(0,mean_control['cum_sum'].max(), 100)

#create column in original df for bins based on cumulative distance
mean_control['bins'] = pd.cut(mean_control['cum_sum'], bins, include_lowest=True)

#compute mean x, mean y  for every bin 
mean_control['meanx'] = mean_control.groupby(['bins'])['x-pos'].transform('mean')
mean_control['meany'] = mean_control.groupby(['bins'])['y-pos'].transform('mean')

mean_control = mean_control [['bins', 'meanx', 'meany']]
mean_control = mean_control.drop_duplicates(inplace=False)
df_ntu_mtway = pd.merge(df_ntu_mtway, mean_control, on="bins", how = "inner")
b = df_ntu_mtway[:100000]
