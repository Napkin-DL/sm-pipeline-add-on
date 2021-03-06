import collections

import pandas as pd
import numpy as np
import os

from io import StringIO

from sklearn.model_selection import train_test_split

import torch


feature_columns_names = [
    'vehicle_id',
    'period_ms',
    'target',
    'make',
    'model',
    'year',
    'vehicle_class',
    'engine_type',
    'voltage_0',
    'voltage_1',
    'voltage_2',
    'voltage_3',
    'voltage_4',
    'voltage_5',
    'voltage_6',
    'voltage_7',
    'voltage_8',
    'voltage_9',
    'voltage_10',
    'voltage_11',
    'voltage_12',
    'voltage_13',
    'voltage_14',
    'voltage_15',
    'voltage_16',
    'voltage_17',
    'voltage_18',
    'voltage_19',
    'current_0',
    'current_1',
    'current_2',
    'current_3',
    'current_4',
    'current_5',
    'current_6',
    'current_7',
    'current_8',
    'current_9',
    'current_10',
    'current_11',
    'current_12',
    'current_13',
    'current_14',
    'current_15',
    'current_16',
    'current_17',
    'current_18',
    'current_19',
] 

class PMDataset_torch:
    '''
    Pytorch dataset to read the AWS fleet predict data.
    '''
    def __init__(self, file, sensor_headers, target_column, standardize=True, verbose=True):      
        super().__init__()
        self.target_column = target_column
        self.sensor_headers = sensor_headers
        self.verbose = verbose
        
        if self.verbose:
            print('Creating PMDataset:', file)
            print('  * Loading CSV data')

        if os.path.isfile(file):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(StringIO(file), header=None)
            df.columns = feature_columns_names
        df = df.dropna()
        
        self.vehicle_properties_headers = self._get_vehicle_properties_headers(df)
        
        self._check_sensor_headers(df)
                
        vehicle_properties_df, sensor_df = self._split_df_columns(df)
        
        mean_dict = self._get_means(sensor_df)
        
        self.data, self.labels = self._build_sensor_output_data(sensor_df, standardize, mean_dict)
        self.vehicle_properties = vehicle_properties_df
        if self.verbose:
            print("Done")
        
    def select_data(self, **kwargs):
        '''
        Function to select data.
        
        Parameters:
        -----------
        **kwargs: {}
            Dictionary where keys are the column names and values are the target selection.
        
        Returns:
        --------
        named_tuple with data, labels, and vehicle_properties.
        
        Usage:
        ------
        To select freightliners,
        
        > train_ds.select_data(make="FREIGHTLINER")
        
        To select 2016 freightliners that use diesel
        
        > train_ds.select_data(make="FREIGHTLINER", model_year=2016, fuel_type="diesel")
        '''
        selection_indexes = np.ones(shape=(self.vehicle_properties.shape[0]))
        for header, selection in kwargs.items():
            selection_index = self.vehicle_properties[header] == selection
            selection_indexes = np.logical_and(selection_indexes, selection_index)
        
        selected_data = self.data[selection_indexes]
        selected_labels = self.labels[selection_indexes]
        selected_vehicle_properties = self.vehicle_properties[selection_indexes]
        Selected_data = collections.namedtuple('Selected_data', 'data labels vehicle_properties')
        return Selected_data(data=selected_data, labels=selected_labels, 
                             vehicle_properties=selected_vehicle_properties)
            
    def _check_sensor_headers(self, df):
        for sensor_header in self.sensor_headers:
            assert df.columns.str.contains(sensor_header).any(), "df does not contain header {}".format(sensor_header)
        
    def _split_train_test_df(self, df):
        # Randomly split df into train and test sets
        df_train, df_test = train_test_split(df, test_size=self.test_size, random_state=12345)
        return df_train, df_test
    
    def _get_vehicle_properties_headers(self, df):
        vehicle_properties = []
        for col in df.columns:
            col = col.strip()
            col_sensor_header = False
            for sensor_header in self.sensor_headers:
                if sensor_header in col:
                    col_sensor_header = True
            if not col_sensor_header:
                vehicle_properties.append(col)
#         print(f"vehicle_properties : {vehicle_properties}")
#         print(f"self.target_column : {self.target_column}")
        vehicle_properties.remove(self.target_column)
        return vehicle_properties
        
    def _split_df_columns(self, df):
        # Helper function to split the data to two dfs, vehicle properties and sensor data.
        vehicle_properties_df = df[self.vehicle_properties_headers]
        sensor_df = df.drop(columns=self.vehicle_properties_headers, errors='ignore')
        return vehicle_properties_df, sensor_df
    
    def _get_means(self, df_train):
        # Calculate the means and std for each header
        output_means = {}
        for sensor_header in self.sensor_headers:
            mean = np.mean(df_train.iloc[:, df_train.columns.str.contains(sensor_header)].values)
            std = np.std(df_train.iloc[:, df_train.columns.str.contains(sensor_header)].values)
            if self.verbose:
                print("{} mean is {:0.4f}+{:0.4f}".format(sensor_header, mean, std))
            output_means[sensor_header] = (mean, std)
        return output_means
    
    def _build_sensor_output_data(self, df, should_standardize, mean_dict):
        labels = df['target']
        
        df = df.drop(columns=['target'],
            errors='ignore')   
        
        data = []
        for sensor_header in self.sensor_headers:
            data_i = df.iloc[:, df.columns.str.contains(sensor_header)].values
            if should_standardize:
                mean, std = mean_dict[sensor_header]
                data_i = (data_i - mean)/std
            data.append(data_i)
        
        return np.array(data).transpose((1,2,0)), labels.values
            
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label
        
    def __len__(self):
        return len(self.data)