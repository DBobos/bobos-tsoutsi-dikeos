# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:32:47 2021

@author: Παναγιώτης
"""
import numpy as np
import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport
#https://anaconda.org/conda-forge/pandas-profiling


# df=pd.read_csv('timeseries_data.csv', delimiter = ',')

#WHEN THE FILE IS TOO BIG SET minimal = True. THIS DOESN'T PRODUCE THE CORRELATION MATRIX
# dff = ProfileReport(df, minimal=True)
# dff.to_file(output_file='output_ser.html')

df=pd.read_csv('timeseries_data.csv', delimiter = ',')

# PROFILE FOR NTU OVERALL
df_ntu = pd.DataFrame()
df_ntu = df.loc[df['location'] != 'HMU']

ntu = ProfileReport(df_ntu)
ntu.to_file('ntu.html')

#PROFILE FOR NTU CONTROL
df_ntu_control = pd.DataFrame()
df_ntu_control = df_ntu.loc[df_ntu['contestant_class'] != 'Patient']

ntu_control = ProfileReport(df_ntu_control)
ntu_control.to_file('ntu_control.html')

#PROFILE FOR NTU CONTROL Motorway

df_ntu_control_mt = df_ntu_control.loc[df_ntu['track'] != 'Urban Low Traffic']

ntu_control = ProfileReport(df_ntu_control_mt)
ntu_control.to_file('ntu_control_motorway.html')

#PROFILE FOR NTU CONTROL Urban Low traffic

df_ntu_control_urb = df_ntu_control.loc[df_ntu['track'] != 'Motorway']

ntu_control = ProfileReport(df_ntu_control_urb)
ntu_control.to_file('ntu_control_urban.html')


#PROFILE FOR NTU PATIENT
df_ntu_patient = pd.DataFrame()
df_ntu_patient = df_ntu.loc[df_ntu['contestant_class'] != 'Control']

ntu_patient = ProfileReport(df_ntu_patient)
ntu_patient.to_file('ntu_patient.html')

#PROFILE FOR NTU PATIENT Motorway
df_ntu_patient_mt = df_ntu_patient.loc[df_ntu['track'] != 'Urban Low Traffic']

ntu_patient = ProfileReport(df_ntu_patient_mt)
ntu_patient.to_file('ntu_patient_motorway.html')

#PROFILE FOR NTU CONTROL Urban Low traffic

df_ntu_patient_urb = df_ntu_patient.loc[df_ntu['track'] != 'Motorway']

ntu_patient = ProfileReport(df_ntu_patient_urb)
ntu_patient.to_file('ntu_patient_urban.html')





























# PROFILE FOR HMU OVERALL
df_hmu = pd.DataFrame()
df_hmu = df.loc[df['location'] != 'NTU']

hmu = ProfileReport(df_hmu)
hmu.to_file('hmu.html')
#PROFILE FOR HMU CONTROL
df_hmu_control = pd.DataFrame()
df_hmu_control = df_hmu.loc[df_hmu['contestant_class'] != 'Patient']

hmu_control = ProfileReport(df_hmu_control)
hmu_control.to_file('hmu_control.html')

#PROFILE FOR HMU CONTROL Motorway

df_hmu_control_mt = df_hmu_control.loc[df_hmu['track'] != 'Urban Low Traffic']

hmu_control = ProfileReport(df_hmu_control_mt)
hmu_control.to_file('hmu_control_motorway.html')

#PROFILE FOR HMU CONTROL Urban Low traffic

df_hmu_control_urb = df_hmu_control.loc[df_hmu['track'] != 'Motorway']

hmu_control = ProfileReport(df_hmu_control_urb)
hmu_control.to_file('hmu_control_urban.html')


#PROFILE FOR HMU PATIENT
df_hmu_patient = pd.DataFrame()
df_hmu_patient = df_hmu.loc[df_hmu['contestant_class'] != 'Control']

hmu_patient = ProfileReport(df_hmu_patient)
hmu_patient.to_file('hmu_patient.html')


#PROFILE FOR HMU PATIENT Motorway
df_hmu_patient_mt = df_hmu_patient.loc[df_hmu['track'] != 'Urban Low Traffic']

hmu_patient = ProfileReport(df_ntu_patient_mt)
hmu_patient.to_file('hmu_patient_motorway.html')

#PROFILE FOR HMU CONTROL Urban Low traffic

df_hmu_patient_urb = df_hmu_patient.loc[df_hmu['track'] != 'Motorway']

hmu_patient = ProfileReport(df_hmu_patient_urb)
hmu_patient.to_file('hmu_patient_urban.html')




df_extra=pd.read_csv('Extra-data.csv', delimiter = ',')


extra = ProfileReport(df_extra)
extra.to_file('extra.html')

# extra control


extra_control = pd.DataFrame()
extra_control = df_extra.loc[df_extra['CASE'] != 1]

control = ProfileReport(extra_control)
control.to_file('extra_control.html')

# extra patient


extra_patient = pd.DataFrame()
extra_patient = df_extra.loc[df_extra['CASE'] != 0]

patient = ProfileReport(extra_control)
patient.to_file('extra_patient.html')



