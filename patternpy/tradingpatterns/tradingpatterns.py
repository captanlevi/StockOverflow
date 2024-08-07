import pandas as pd
import numpy as np


def detect_head_shoulder(df, window=3):
# Define the rolling window
    roll_window = window
    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    # Create a boolean mask for Head and Shoulder pattern
    mask_head_shoulder = ((df['high_roll_max'] > df['high'].shift(1)) & (df['high_roll_max'] > df['high'].shift(-1)) & (df['high'] < df['high'].shift(1)) & (df['high'] < df['high'].shift(-1)))
    # Create a boolean mask for Inverse Head and Shoulder pattern
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['low'].shift(1)) & (df['low_roll_min'] < df['low'].shift(-1)) & (df['low'] > df['low'].shift(1)) & (df['low'] > df['low'].shift(-1)))
    # Create a new column for Head and Shoulder and its inverse pattern and populate it using the boolean masks
    df['head_shoulder_pattern'] = 0
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 1
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = -1
    return df 
    # return not df['head_shoulder_pattern'].isna().any().item()

def detect_multiple_tops_bottoms(df, window=3):
# Define the rolling window
    roll_window = window
    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['close'].rolling(window=roll_window).min()
    # Create a boolean mask for multiple top pattern
    mask_top = (df['high_roll_max'] >= df['high'].shift(1)) & (df['close_roll_max'] < df['close'].shift(1))
    # Create a boolean mask for multiple bottom pattern
    mask_bottom = (df['low_roll_min'] <= df['low'].shift(1)) & (df['close_roll_min'] > df['close'].shift(1))
    # Create a new column for multiple top bottom pattern and populate it using the boolean masks
    df['multiple_top_bottom_pattern'] = 0
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 1
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = -1
    return df

def calculate_support_resistance(df, window=3):
# Define the rolling window
    roll_window = window
    # Set the number of standard deviation
    std_dev = 2
    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    # Calculate the mean and standard deviation for high and low
    mean_high = df['high'].rolling(window=roll_window).mean()
    std_high = df['high'].rolling(window=roll_window).std()
    mean_low = df['low'].rolling(window=roll_window).mean()
    std_low = df['low'].rolling(window=roll_window).std()
    # Create a new column for support and resistance
    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high
    return df
def detect_triangle_pattern(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    # Create a boolean mask for ascending triangle pattern
    mask_asc = (df['high_roll_max'] >= df['high'].shift(1)) & (df['low_roll_min'] <= df['low'].shift(1)) & (df['close'] > df['close'].shift(1))
    # Create a boolean mask for descending triangle pattern
    mask_desc = (df['high_roll_max'] <= df['high'].shift(1)) & (df['low_roll_min'] >= df['low'].shift(1)) & (df['close'] < df['close'].shift(1))
    # Create a new column for triangle pattern and populate it using the boolean masks
    df['triangle_pattern'] = 0
    df.loc[mask_asc, 'triangle_pattern'] = 1
    df.loc[mask_desc, 'triangle_pattern'] = -1
    return df

def detect_wedge(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    df['trend_high'] = df['high'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    # Create a boolean mask for Wedge Up pattern
    mask_wedge_up = (df['high_roll_max'] >= df['high'].shift(1)) & (df['low_roll_min'] <= df['low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Wedge Down pattern
        # Create a boolean mask for Wedge Down pattern
    mask_wedge_down = (df['high_roll_max'] <= df['high'].shift(1)) & (df['low_roll_min'] >= df['low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Wedge Up and Wedge Down pattern and populate it using the boolean masks
    df['wedge_pattern'] = 0
    df.loc[mask_wedge_up, 'wedge_pattern'] = 1
    df.loc[mask_wedge_down, 'wedge_pattern'] = -1
    return df
def detect_channel(df, window=3):
    # Define the rolling window
    roll_window = window
    # Define a factor to check for the range of channel
    channel_range = 0.1
    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    df['trend_high'] = df['high'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    # Create a boolean mask for Channel Up pattern
    mask_channel_up = (df['high_roll_max'] >= df['high'].shift(1)) & (df['low_roll_min'] <= df['low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Channel Down pattern
    mask_channel_down = (df['high_roll_max'] <= df['high'].shift(1)) & (df['low_roll_min'] >= df['low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Channel Up and Channel Down pattern and populate it using the boolean masks
    df['channel_pattern'] = 0
    df.loc[mask_channel_up, 'channel_pattern'] = 1
    df.loc[mask_channel_down, 'channel_pattern'] = -1
    return df

def detect_double_top_bottom(df, window=3, threshold=0.05):
    # Define the rolling window
    roll_window = window
    # Define a threshold to check for the range of pattern
    range_threshold = threshold

    # Create a rolling window for high and low
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()

    # Create a boolean mask for Double Top pattern
    mask_double_top = (df['high_roll_max'] >= df['high'].shift(1)) & (df['high_roll_max'] >= df['high'].shift(-1)) & (df['high'] < df['high'].shift(1)) & (df['high'] < df['high'].shift(-1)) & ((df['high'].shift(1) - df['low'].shift(1)) <= range_threshold * (df['high'].shift(1) + df['low'].shift(1))/2) & ((df['high'].shift(-1) - df['low'].shift(-1)) <= range_threshold * (df['high'].shift(-1) + df['low'].shift(-1))/2)
    # Create a boolean mask for Double Bottom pattern
    mask_double_bottom = (df['low_roll_min'] <= df['low'].shift(1)) & (df['low_roll_min'] <= df['low'].shift(-1)) & (df['low'] > df['low'].shift(1)) & (df['low'] > df['low'].shift(-1)) & ((df['high'].shift(1) - df['low'].shift(1)) <= range_threshold * (df['high'].shift(1) + df['low'].shift(1))/2) & ((df['high'].shift(-1) - df['low'].shift(-1)) <= range_threshold * (df['high'].shift(-1) + df['low'].shift(-1))/2)

    # Create a new column for Double Top and Double Bottom pattern and populate it using the boolean masks
    df['double_pattern'] = 0
    df.loc[mask_double_top, 'double_pattern'] = 1
    df.loc[mask_double_bottom, 'double_pattern'] = -1

    return df

def detect_trendline(df, window=2):
    # Define the rolling window
    roll_window = window
    # Create new columns for the linear regression slope and y-intercept
    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df['close'][i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    # Create a boolean mask for trendline support
    mask_support = df['slope'] > 0

    # Create a boolean mask for trendline resistance
    mask_resistance = df['slope'] < 0

    # Create new columns for trendline support and resistance
    df['support'] = np.nan
    df['resistance'] = np.nan

    # Populate the new columns using the boolean masks
    df.loc[mask_support, 'support'] = df['close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['close'] * df['slope'] + df['intercept']

    return df

def find_pivots(df):
    # Calculate differences between consecutive highs and lows
    high_diffs = df['high'].diff()
    low_diffs = df['low'].diff()

    # Find higher high
    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    
    # Find lower low
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)

    # Find lower high
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)

    # Find higher low
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    # Create signals column
    df['signal'] = 0
    df.loc[higher_high_mask, 'signal'] = -2#'HH'
    df.loc[lower_low_mask, 'signal'] = -1#'LL'
    df.loc[lower_high_mask, 'signal'] = 1#'LH'
    df.loc[higher_low_mask, 'signal'] = 2#'HL'

    return df