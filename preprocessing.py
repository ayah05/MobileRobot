import pandas as pd
import numpy as np

# function to load data from csv
def load_data(input_file, delimiter=',', decimal='.'):
    """
    load robot motion data from a csv file generated using ROS
    """
    try:
        raw_data = pd.read_csv('raw_data/mir100_sample_data.csv', delimiter=delimiter, decimal=decimal)
        return raw_data
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


# function to clean and preprocess data
def clean_data(data):
    
    
    return cleaned_data
    


# function to split data into training, validation and test datasets
def split_data(data):
    
    return train_data, validation_data, test_data



# function to save the preprocessed data for later use by the ML model
def save_data(train_data, validation_data, test_data, output_directory, format='PyTorch'):
    
    return train_data_file, validation_data_file, test_data_file 
    
    


# Data Preprocessing Main Function
def preprocess_data (input_file, output_directory, format='PyTorch'):
    raw_data = load_data('/raw_data/mir100_sample_data.csv', delimiter=",", decimal='.')
    cleaned_data = load_data(raw_data)
    train_data, validation_data, test_data = split_data(cleaned_data)
    train_data_file, validation_data_file, test_data_file =  save_data(train_data, validation_data, test_data, output_directory, format=format)


# test-drive the execution
preprocess_data("input.csv", "preprocessed_data/", format='PyTorch')
    
    