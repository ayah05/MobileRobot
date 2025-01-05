import pandas as pd
import numpy as np
import torch

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
def clean_data(data, timestep_interval):
    """
    clean the data by removing outliers and by interpolating missing values
    
    Parameters
    ----------
    data : np array
        raw mobile robot motion data
    timestep_interval : float
        interval between two timesteps in seconds

    Returns
    -------
    np array
        numpy array of cleaned data
    """

    
    len_before = len(data)

    # remove rows with extreme velocities
    linear_velocity_threshold = 1.5  # m/s, threshold for outlier removal (according to MiR100 datasheet)
    cleaned_data = data[(data['linear_velocity'] <= linear_velocity_threshold)]
    
    # remove rows with extreme angular velocities
    mean_value = data['angular_velocity'].mean()
    std_value = data['angular_velocity'].std()
    data = data[(data['angular_velocity'] >= mean_value - 3 * std_value) & (data['angular_velocity'] <= mean_value + 3 * std_value)]
    
    len_after = len(data)
    removed_lines = len_before - len_after
    print(removed_lines, "outlier lines were removed")
    
    data['timestamp'] = pd.to_datetime(data['timestamp']) # convert timestamp column to datetime format
    data = data.sort_values(by='timestamp').reset_index(drop=True) # ensure timestamps are sorted
    
    # create a complete timeline without any missing timestamp
    min_time = data['timestamp'].min()
    max_time = data['timestamp'].max()
    complete_timeline = pd.date_range(start=min_time, end=max_time, freq=f'{timestep_interval}S')
    
    # reindex data to match the complete timeline
    data = data.set_index('timestamp')
    data = data.reindex(complete_timeline, method=None) # this will insert NaN values for missing timestamps
    data.index.name = 'timestamp'  # ensure index remains labeled as 'timestamp'

    # interpolate missing values and make the timestamp column a normal column again
    data = data.interpolate(method='linear', limit_direction='both').reset_index()
    

    return cleaned_data


# function to convert quaternions into Euler angles
def quaternion_to_euler(w, x, y, z):
    """
    convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z
    

# function to generate state-action-sequences
def generate_state_action_sequences(data, sequence_length, output_size):
    """
    generate sequences of state-action pairs
    the first element in a sequence represents the earliest timestep in the sequence,
    the last element in a sequence represents the most recent timestep in the sequence.
    """
    # State: positions and orientations
    data['roll'], data['pitch'], data['yaw'] = zip(*data.apply(lambda row: quaternion_to_euler(row['orientation_w'],
                                                                                            row['orientation_x'],
                                                                                            row['orientation_y'],
                                                                                            row['orientation_z']), axis=1))
    states = data[['position_x', 'position_y', 'roll', 'pitch', 'yaw']].values

    # Action: linear and angular velocities
    actions = data[['linear_velocity', 'angular_velocity']].values

    # Create sequences
    state_sequences = []
    action_sequences = []

    for i in range(len(states) - sequence_length - output_size + 1):
        state_sequences.append(states[i:i + sequence_length])
        action_sequences.append(actions[i + sequence_length:i + sequence_length + output_size])

    return np.array(state_sequences), np.array(action_sequences)


# function to split data into training, validation and test datasets
def split_data(states, actions, val_size=0.1, test_size=0.2):
    """
    split the dataset into training, validation, and test sets - no shuffling
    """
    train_size = 1 - val_size - test_size
    train_length = int(train_size * len(states))
    val_length = int(val_size * len(states))
        
    train_states=states[:train_length]
    train_actions=actions[:train_length]
    val_states=states[train_length:train_length+val_length]
    val_actions=actions[train_length:train_length+val_length]
    test_states=states[train_length+val_length:]
    test_actions=actions[train_length+val_length:]
    
    return train_states, val_states, test_states, train_actions, val_actions, test_actions


# function to save the preprocessed data for later use by the ML model
def save_data(train_states, val_states, test_states, train_actions, val_actions, test_actions, output_directory, save_as_pt=True):
    """
    save the datasets in PyTorch's pt format, or csv, respectively
    """
    if save_as_pt:
        torch.save({'states': train_states, 'actions': train_actions}, f'{output_directory}/train.pt')
        torch.save({'states': val_states, 'actions': val_actions}, f'{output_directory}/val.pt')
        torch.save({'states': test_states, 'actions': test_actions}, f'{output_directory}/test.pt')
    else:
        pd.DataFrame({'states': list(train_states), 'actions': list(train_actions)}).to_csv(f'{output_directory}/train.csv', index=False)
        pd.DataFrame({'states': list(val_states), 'actions': list(val_actions)}).to_csv(f'{output_directory}/val.csv', index=False)
        pd.DataFrame({'states': list(test_states), 'actions': list(test_actions)}).to_csv(f'{output_directory}/test.csv', index=False)    


# Data Preprocessing Main Function
def preprocess_data (input_file, output_directory, save_as_pt=False, timestep_interval=1, sequence_length=10, output_size=1, val_size=0.1, test_size=0.2):
    """
    preprocesses motion data from a CSV file, including cleaning, interpolation, 
    formatting into state-action sequences, and splitting into train, validation, and test datasets. 
    The datasets can be saved as .pt files or .csv files.

    Parameters
    ----------
    input_file : str
        path to the input CSV file containing raw motion data
    output_directory : str
        directory where the preprocessed datasets will be saved
    save_as_pt : bool, optional
        whether to save the datasets as .pt files for PyTorch, by default False - then the file is saved as CSV
    timestep_interval : int, optional
        interval (in seconds) for ensuring consistent timestamps in the data, by default 1
    sequence_length : int, optional
        number of timesteps to include in each input sequence, by default 10.
    output_size : int, optional
        number of timesteps to predict for each output sequence, by default 1.
    val_size : float, optional
        proportion of the data to use for validation, by default 0.1.
    test_size : float, optional
        proportion of the data to use for testing, by default 0.2.
    """
    
    raw_data = load_data('/raw_data/mir100_sample_data.csv', delimiter=",", decimal='.')
    cleaned_data = clean_data(raw_data, timestep_interval=timestep_interval)
    states, actions = generate_state_action_sequences(cleaned_data, sequence_length=sequence_length, output_size=output_size)
    train_states, val_states, test_states, train_actions, val_actions, test_actions = split_data(states, actions, val_size=val_size, test_size=test_size)
    save_data(train_states, val_states, test_states, train_actions, val_actions, test_actions, output_directory, save_as_pt=save_as_pt)


# test-drive the execution
preprocess_data("input.csv", output_directory="preprocessed_data/", save_as_pt=True, timestep_interval=1, sequence_length=5, output_size=1, val_size=0.1, test_size=0.2)
    
    