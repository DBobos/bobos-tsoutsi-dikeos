# import libraries
import os
from glob import glob
import pandas as pd
import re


# get path of current working directory
# ! it is important to set current working directory to the folder
# ! where the script and data folder are located
cwd = os.getcwd()


def mean_xlsx(filepath):
    """
    takes a path to an xlsx file, reads it and returns mean column values
    """
    # read xlsx data file
    data = pd.read_excel(filepath)
    
    # locate the first empty row
    first_empty_row = data[data.isnull().all(axis=1) == True].index.tolist()[0]
    
    # remove everything from this line and below
    data = data.loc[0:first_empty_row-1]
    
    # drop the columns where all elements are NaN
    data = data.dropna(axis=1, how='all')
    
    # create a dataframe to aggregate data and store averages
    df = pd.DataFrame()
    
    # calculate averages for each column
    for column in data:
        df[column] = [data[column].mean()]
    return df


def mean_txt(filepath):
    """
    takes a path to a txt file, reads it and returns mean column values
    """
    # read txt data file
    data = pd.read_csv(filepath,
                       delimiter = '\t',
                       usecols=[*range(0,28)],
                       skiprows=2
                       )
    
    # create a dataframe to aggregate data and store averages
    df = pd.DataFrame()
    
    # calculate averages for each column
    for column in data:
        df[column] = [data[column].mean()]
    return df


def load_files(directory, pattern):
    """
    takes as input a directory and a pattern and returns a dataframe with the
    average column values for all the files in the directory and its
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
    
    # create a dictionary to store the class for each simulator data file,
    # based on the filename (classes: control / patient)
    files_classified = dict()
    for file in files:
        if re.match(r'.*Simulator\s((?:NTU)|(?:HMU))\\Controls\\.*', file):
            files_classified[file] = 'Control'
        elif re.match(r'.*Simulator\s((?:NTU)|(?:HMU))\\Patients\\.*', file):
            files_classified[file] = 'Patient'
    
    # iterate through files and generate a dataframe with average col values
    df = pd.DataFrame(columns=['class'])
    for file in files_classified.keys():
        if (file.endswith('.xlsx')) or (file.endswith('.xls')):
            df = df.append(mean_xlsx(file), ignore_index=True)
        elif file.endswith('.txt'):
            df = df.append(mean_txt(file), ignore_index=True)
    df['class'] = files_classified.values()
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

# generate hmu urban-low-traffic simulator dataframe
df_hmu_urblt = load_files(hmu_directory, hmu_urblt_pattern)

# generate ntu motorway simulator dataframe
df_ntu_mtway = load_files(ntu_directory, ntu_mtway_pattern)

# generate ntu urban-low-traffic simulator dataframe
df_ntu_urblt = load_files(ntu_directory, ntu_urblt_pattern)

