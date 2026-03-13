# import libraries
import os
from glob import glob
import pandas as pd
import numpy as np
import re
from functools import reduce



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



"""
GENERATE DATAFRAMES
"""

# generate hmu motorway simulator dataframe
df_hmu_mtway = load_files(hmu_directory, hmu_mtway_pattern)
# conversions
df_hmu_mtway['foll_dist_in_seconds'] = df_hmu_mtway['foll_dist_in_seconds'].replace(-1, np.nan)  # not to be aggregated
df_hmu_mtway['lane_offset (m)'] = np.abs(df_hmu_mtway['lane_offset (m)']) # keep absolute value in meters
# add track label
df_hmu_mtway['track'] = 'Motorway'
# select columns
df_hmu_mtway = df_hmu_mtway[['contestant_class',
                             'contestant',
                             'location',
                             'track',
                             'event_class',
                             'speed (m/s)',
                             'accelerator',
                             'gear',
                             'brake',
                             'steering',
                             'foll_dist_in_seconds',
                             'lane_offset (m)',
                             'time'
                             ]]
# rename columns
df_hmu_mtway = df_hmu_mtway.rename(columns={'speed (m/s)': 'speed_mps',
                                            'foll_dist_in_seconds': 'headway_s',
                                            'lane_offset (m)': 'lane_offset_m',
                                            'time': 'time_s'
                                            }
                                   )



# generate hmu urban-low-traffic simulator dataframe
df_hmu_urblt = load_files(hmu_directory, hmu_urblt_pattern)
# conversions
df_hmu_urblt['HWay'] = df_hmu_urblt['current_time_headway'].replace(-1, np.nan)  # not to be aggregated
# add track label
df_hmu_urblt['track'] = 'Urban Low Traffic'
# select columns
df_hmu_urblt = df_hmu_urblt[['contestant_class',
                             'contestant',
                             'location',
                             'track',
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
                             'pos_z',
                             'dt'
                             ]]
# rename columns
df_hmu_urblt = df_hmu_urblt.rename(columns={'current_speed': 'speed_mps',
                                            'accel': 'accelerator',
                                            'current_gear': 'gear',
                                            'brake_pedal_posn': 'brake',
                                            'steering_posn': 'steering',
                                            'current_time_headway': 'headway_s'
                                            }
                                   )



# generate ntu motorway simulator dataframe
df_ntu_mtway = load_files(ntu_directory, ntu_mtway_pattern)
# conversions
df_ntu_mtway['Speed'] = (1000 / 3600) * df_ntu_mtway['Speed'] # km/h -> m/s
df_ntu_mtway['Time'] = df_ntu_mtway['Time'] / 1000  # ms -> s
df_ntu_mtway['HWay'] = df_ntu_mtway['HWay'].replace(9999.9, np.nan)  # not to be aggregated
# add track label
df_ntu_mtway['track'] = 'Motorway'
# select columns
df_ntu_mtway = df_ntu_mtway[['contestant_class',
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
df_ntu_mtway = df_ntu_mtway.rename(columns={'Speed': 'speed_mps',
                                            'Acc': 'accelerator',
                                            'Gear': 'gear',
                                            'Brk': 'brake',
                                            'Wheel': 'steering',
                                            'HWay': 'headway_s',
                                            'ralpha': 'orientation',
                                            'x-pos': 'pos_x',
                                            'y-pos': 'pos_y',
                                            'z-pos': 'pos_z',
                                            'Time': 'time_s'
                                            }
                                   )



# generate ntu urban-low-traffic simulator dataframe
df_ntu_urblt = load_files(ntu_directory, ntu_urblt_pattern)
# conversions
df_ntu_urblt['Speed'] = (1000 / 3600) * df_ntu_urblt['Speed']  # km/h -> m/s
df_ntu_urblt['Time'] = df_ntu_urblt['Time'] / 1000  # ms -> s
df_ntu_urblt['HWay'] = df_ntu_urblt['HWay'].replace(9999.9, np.nan)  # not to be aggregated
# add track label
df_ntu_urblt['track'] = 'Urban Low Traffic'
# select columns
df_ntu_urblt = df_ntu_urblt[['contestant_class',
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
df_ntu_urblt = df_ntu_urblt.rename(columns={'Speed': 'speed_mps',
                                            'Acc': 'accelerator',
                                            'Gear': 'gear',
                                            'Brk': 'brake',
                                            'Wheel': 'steering',
                                            'HWay': 'headway_s',
                                            'ralpha': 'orientation',
                                            'x-pos': 'pos_x',
                                            'y-pos': 'pos_y',
                                            'z-pos': 'pos_z',
                                            'Time': 'time_s'
                                            }
                                   )



# calculate instant acceleration for df_hmu_mtway
df_hmu_mtway['acceleration'] = df_hmu_mtway.groupby(['contestant'])['speed_mps'].diff() \
                               / df_hmu_mtway.groupby(['contestant'])['time_s'].diff()

# calculate distance travelled by each contestant for df_hmu_mtway
df_hmu_mtway['distance'] = (df_hmu_mtway['speed_mps'].shift() * df_hmu_mtway.groupby(['contestant'])['time_s'].diff()) \
       + (df_hmu_mtway['acceleration'] * (df_hmu_mtway.groupby(['contestant'])['time_s'].diff() ** 2) / 2)

# calculate time passed from dt for df_hmu_mtway
df_hmu_urblt['time_s'] = df_hmu_urblt.assign(dt=pd.to_numeric(df_hmu_urblt['dt'], errors='coerce')) \
    .groupby(['contestant'])['dt'].cumsum(axis=0)

# calculate instant acceleration for df_hmu_urblt
df_hmu_urblt['acceleration'] = df_hmu_urblt.groupby(['contestant'])['speed_mps'].diff() \
                               / df_hmu_urblt.groupby(['contestant'])['time_s'].diff()

# calculate distance travelled by each contestant for df_hmu_urblt
df_hmu_urblt['distance'] = (df_hmu_urblt['speed_mps'].shift() * df_hmu_urblt.groupby(['contestant'])['time_s'].diff()) \
       + (df_hmu_urblt['acceleration'] * (df_hmu_urblt.groupby(['contestant'])['time_s'].diff() ** 2) / 2)

# calculate instant acceleration for df_hmu_mtway
df_ntu_urblt['acceleration'] = df_ntu_urblt.groupby(['contestant'])['speed_mps'].diff() \
                               / df_ntu_urblt.groupby(['contestant'])['time_s'].diff()

# calculate distance travelled by each contestant for df_ntu_urblt
df_ntu_urblt['distance'] = (df_ntu_urblt['speed_mps'].shift() * df_ntu_urblt.groupby(['contestant'])['time_s'].diff()) \
       + (df_ntu_urblt['acceleration'] * (df_ntu_urblt.groupby(['contestant'])['time_s'].diff() ** 2) / 2)

# calculate instant acceleration for df_ntu_mtway
df_ntu_mtway['acceleration'] = df_ntu_mtway.groupby(['contestant'])['speed_mps'].diff() \
                               / df_ntu_mtway.groupby(['contestant'])['time_s'].diff()

# calculate distance travelled by each contestant for df_ntu_mtway
df_ntu_mtway['distance'] = (df_ntu_mtway['speed_mps'].shift() * df_ntu_mtway.groupby(['contestant'])['time_s'].diff()) \
       + (df_ntu_mtway['acceleration'] * (df_ntu_mtway.groupby(['contestant'])['time_s'].diff() ** 2) / 2)



# is this needed?
df_hmu_mtway['acceleration'] = df_hmu_mtway['acceleration'].replace(np.nan, 0)
df_hmu_mtway['distance'] = df_hmu_mtway['distance'].replace(np.nan, 0)
df_hmu_urblt['acceleration'] = df_hmu_urblt['acceleration'].replace(np.nan, 0)
df_hmu_urblt['distance'] = df_hmu_urblt['distance'].replace(np.nan, 0)
df_ntu_mtway['acceleration'] = df_ntu_mtway['acceleration'].replace(np.nan, 0)
df_ntu_mtway['distance'] = df_ntu_mtway['distance'].replace(np.nan, 0)
df_ntu_urblt['acceleration'] = df_ntu_urblt['acceleration'].replace(np.nan, 0)
df_ntu_urblt['distance'] = df_ntu_urblt['distance'].replace(np.nan, 0)



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
    # df['cumulative_distance'] = df.groupby(['contestant'])['distance'].cumsum(axis=0)
    dist = df.groupby((df.contestant != df.contestant.shift(1)).cumsum())['distance'].cumsum()
    df['cumulative_distance'] = dist
    
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


def positioning_ntu(df):
    """
    takes as input a dataframe with the entries of all contestants on a specific
    track and simulator and returns the mean values of coordinates for each
    contestant in each distance bin uses pos_z for pos_y
    """
    
    # calculate cumulative distance and distance bins
    df = distance_bins(df)

    # calculate mean X and Y values per contestant and distance bin
    df['mean_x'] = df.groupby(['bins', 'contestant_class'])['pos_x'].transform('mean')
    df['mean_y'] = df.groupby(['bins', 'contestant_class'])['pos_z'].transform('mean')
    
    return df


def positioning_hmu(df):
    """
    takes as input a dataframe with the entries of all contestants on a specific
    track and simulator and returns the mean values of coordinates for each
    contestant in each distance bin 
    """
    
    # calculate cumulative distance and distance bins
    df = distance_bins(df)

    # calculate mean X and Y values per contestant and distance bin
    df['mean_x'] = df.groupby(['bins', 'contestant_class'])['pos_x'].transform('mean')
    df['mean_y'] = df.groupby(['bins', 'contestant_class'])['pos_y'].transform('mean')
    
    return df

# calculate route coordinates for df_hmu_urblt
route_hmu_urblt = route_estimation(df_hmu_urblt)

# calculate each contestant's positioning for df_hmu_urblt
df_hmu_urblt = positioning_hmu(df_hmu_urblt)

# get route coordinates per distance bin on df_hmu_urblt
df_hmu_urblt = pd.merge(df_hmu_urblt, route_hmu_urblt, on='bins', how='inner')

# calculate the lane offset in meters as the distance of each contestant's
# position from the route coordinates, for df_hmu_urblt 
df_hmu_urblt['lane_offset_m'] = np.sqrt((df_hmu_urblt['mean_x'] - df_hmu_urblt['route_x'])**2 + \
                                        (df_hmu_urblt['mean_y'] - df_hmu_urblt['route_y'])**2
                                        )

# calculate route coordinates for df_ntu_mtway
route_ntu_mtway = route_estimation(df_ntu_mtway)

# calculate each contestant's positioning for df_ntu_mtway
df_ntu_mtway = positioning_ntu(df_ntu_mtway)

# get route coordinates per distance bin on df_ntu_mtway
df_ntu_mtway = pd.merge(df_ntu_mtway, route_ntu_mtway, on='bins', how='inner')

# calculate the lane offset in meters as the distance of each contestant's
# position from the route coordinates, for df_ntu_mtway
df_ntu_mtway['lane_offset_m'] = np.sqrt((df_ntu_mtway['mean_x'] - df_ntu_mtway['route_x'])**2 + \
                                        (df_ntu_mtway['mean_y'] - df_ntu_mtway['route_y'])**2
                                        )

# calculate route coordinates for df_ntu_urblt
route_ntu_urblt = route_estimation(df_ntu_urblt)

# calculate each contestant's positioning for df_ntu_urblt
df_ntu_urblt = positioning_ntu(df_ntu_urblt)

# get route coordinates per distance bin on df_ntu_urblt
df_ntu_urblt = pd.merge(df_ntu_urblt, route_ntu_urblt, on='bins', how='inner')

# calculate the lane offset in meters as the distance of each contestant's
# position from the route coordinates, for df_ntu_mtway
df_ntu_urblt['lane_offset_m'] = np.sqrt((df_ntu_urblt['mean_x'] - df_ntu_urblt['route_x'])**2 + \
                                        (df_ntu_urblt['mean_y'] - df_ntu_urblt['route_y'])**2
                                        )

# define the list of the desired columns for the urban low traffic dataframe
urblt_columns = ['contestant_class',
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
# for the 'Urban Low Traffic' scenario
urblt_dataframes = [df_hmu_urblt[[c for c in urblt_columns]],
                    df_ntu_urblt[[c for c in urblt_columns]]
                    ]

# create the final dataframe (df_u) for the 'Urban Low Traffic' scenario
df_u = pd.concat(urblt_dataframes, ignore_index=True)



# define the list of the desired columns for the motorway dataframe
mtway_columns = ['contestant_class',
                 'contestant',
                 'location',
                 'track',
                 'event_class',
                 'time_s',
                 'distance',
                 #'cumulative_distance',
                 'speed_mps',
                 'acceleration',
                 'accelerator',
                 'gear',
                 'brake',
                 'steering',
                 'headway_s',
                 'lane_offset_m'
                 ]

# define the list of datasets that will be used to generate the final dataframe
# for the 'Motorway' scenario
mtway_dataframes = [df_hmu_mtway[[c for c in mtway_columns]],
                    df_ntu_mtway[[c for c in mtway_columns]]
                    ]

# create the final dataframe (df_m) for the 'Motorway' scenario
df_m = pd.concat(mtway_dataframes, ignore_index=True)

# calculate mean values for the 'Urban Low Traffic' scenario
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

# calculate standard deviation for the 'Urban Low Traffic' scenario
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

# calculate mean values for the 'Motorway' scenario
df_m_mean = df_m.groupby(['contestant_class',
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
                            'lane_offset_m']].mean()

# calculate standard deviation for the 'Motorway' scenario
df_m_std = df_m.groupby(['contestant_class',
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
                            'lane_offset_m']].std()


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
df_u_mean = rename_columns(df_u_mean, 'urban', 'mean')
df_u_std = rename_columns(df_u_std, 'urban', 'std')
df_m_mean = rename_columns(df_m_mean, 'motorway', 'mean')
df_m_std = rename_columns(df_m_std, 'motorway', 'std')

# create a list of above dataframes in order to merge them
dataframes = [df_u_mean, df_u_std, df_m_mean, df_m_std]

# merge dataframes into one
df_agg = reduce(lambda  left, right: pd.merge(left, right,
                                              on=['contestant_class', 'contestant', 'location' ],
                                              how='outer'), dataframes)

# sort df_agg columns by name
df_agg = df_agg.reindex(sorted(df_agg.columns), axis=1)

# export df_agg to csv
#df_agg.to_csv('aggregated_data.csv', index=True)



