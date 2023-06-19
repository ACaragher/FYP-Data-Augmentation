import pandas as pd
import numpy as np

from random import randrange
import rocket
import mpdatasets
from dtw import *

def reverse_dataframe(dfx):
    reversed_rows = []

    # Loop through each row
    for i in range(dfx.shape[0]):
        body_parts = []

        #Loop through each body part in the row
        for j in range(dfx.shape[1]):
            # appends reversed time series to the body_parts array
            # reset the index of the series as otherwise it gets reversed as well
            body_parts.append(dfx.iloc[i][j][::-1].reset_index(drop=True))
        reversed_rows.append(body_parts) # append all reversed time series for row i of train_x to the reversed_rows array
    
    reverse_frame = pd.DataFrame(reversed_rows)
    reverse_frame.columns = dfx.columns
    return pd.DataFrame(reverse_frame)
    

    
def pad_series(dfx, pad_size, multiplier):
    padded_series_list = []

    for i in range(dfx.shape[0]):

        padded_series = []

        for j in range(dfx.shape[1]):
            
            first_value = dfx.iloc[i][j][0].item()
            final_value = dfx.iloc[i][j][-1:].item()

            num_pad = pad_size * (multiplier-1)
            
            # Creates empty arrays half the window size
            # and fills them with either the first or last value
            # If the window size is odd, add 1 extra to the start
            if num_pad % 2 == 0:
                pad_start = np.empty(num_pad // 2)
            else:
                pad_start = np.empty(num_pad // 2 + 1)
            pad_end = np.empty(num_pad // 2)
            pad_start.fill(first_value)
            pad_end.fill(final_value)
            start_series = pd.Series(pad_start)
            end_series = pd.Series(pad_end)

            padded_timeseries = pd.concat([start_series, dfx.iloc[i][j], end_series], axis=0, ignore_index=True) 
            padded_series.append(padded_timeseries)

        padded_series_list.append(padded_series)
       
        padded_frame = pd.DataFrame(padded_series_list)
        padded_frame.columns = dfx.columns
    return padded_frame



def window_warp(dfx, window_size, multiplier, stepped=False):
    warped_series = []

    ts_length = len(dfx.iloc[0][0])

    for i in range(dfx.shape[0]):

        window_start = ts_length // 2 - window_size // 2

        window_end = window_start + window_size
        body_parts = []
        
        
        for j in range(dfx.shape[1]):
            
            # Add any values form 0 to beginning of the window to the aug_time_series
            # Then, for each value in the window, add it'multiplier' times to aug_time_series
            # Finally, add any values after the window till the end to the aug_time_series
            aug_time_series = dfx.iloc[i][j][:window_start] 
            if stepped:
                for k in range(window_start, window_end):
                    for m in range(0,multiplier):
                        aug_time_series = pd.concat([aug_time_series, dfx.iloc[i][j][k:k+1]], axis=0, ignore_index=True)
                        
                aug_time_series = pd.concat([aug_time_series, dfx.iloc[i][j][window_end:]], axis=0, ignore_index=True)
        
                body_parts.append(aug_time_series)

            # To warp: take two adjacent values. Add 'multiplier' number of new values 
            # between them. The values are linear between the two original values
            else:
                for k in range(window_start, window_end):
                    #
                    difference = dfx.iloc[i][j][k+1].item() - dfx.iloc[i][j][k].item()
                    change_amount = 0
                    # The first value is added to the time series on the first iteration
                    # On subsequent iterations, difference/multiplier is added each time to the
                    # previously added value
                    for m in range(0,multiplier):
                        new_value = pd.Series([dfx.iloc[i][j][k].item() + change_amount])
                        aug_time_series = pd.concat([aug_time_series, new_value], axis=0, ignore_index=True)
                        change_amount += difference/multiplier

                aug_time_series = pd.concat([aug_time_series, dfx.iloc[i][j][window_end:]], axis=0, ignore_index=True)

                body_parts.append(aug_time_series)

        warped_series.append(body_parts)
        
    warped_frame = pd.DataFrame(warped_series)
    warped_frame.columns = dfx.columns
    return warped_frame



def dtw_interpolate(df_x, df_y, randomise=False):
    
    frame_x = df_x.copy()
    frame_x['Label'] = df_y.tolist()
    
    aug_x = pd.DataFrame()
    aug_y = []
    
    # Group dataframe by label, then pass each grouped dataframe
    for label, label_df in frame_x.groupby(by=['Label']):
        temp_x = dtw_generate(label_df.iloc[:,:-1], randomise)
        
        # Returned df temp_x should all be assigned the label of the dataframe 
        # Creates array of length 'temp_x' so it matches the values in temp_x
        aug_y = np.append(aug_y, [label] * len(temp_x))
        aug_x = pd.concat([aug_x, temp_x], axis=0, ignore_index=True)

    return aug_x, aug_y



# Aligns with DTW then produces interpolated time series
def dtw_generate(dfx, randomise):
    
    aug_df = []
    if randomise:
        #Randomises order of rows
        dfx = dfx.sample(frac=1).reset_index(drop=True) 
    
    for i in range(1, dfx.shape[0]):
        
        body_parts = []
        
        # For each row, align it with the previous row
        # Use the returned arrays to match the inices in the two time series
        # and get the mean of the two or 3 values
        for j in range(dfx.shape[1]):
            alignment = dtw(dfx.iloc[i-1][j], dfx.iloc[i][j], keep_internals=True,  step_pattern=rabinerJuangStepPattern(2,"c"))
            warp_a = alignment.index1
            warp_b = alignment.index2

            new_timeseries = []
            
            # Loop through all values in the arrays
            # If a value is repeated in the array, that means that it is aligned with 2 points 
            # in the other series
            # Otherwise it is just the mean between the two aligned values 
            m = 1
            while m < len(warp_a):
                if(warp_a[m] == warp_a[m-1]):
                    interpolated_value = dfx.iloc[i-1][j][warp_a[m-1]] + dfx.iloc[i][j][warp_b[m-1]] + dfx.iloc[i][j][warp_b[m]]
                    interpolated_value /= 3
                    m += 2
                else:
                    interpolated_value = dfx.iloc[i-1][j][warp_a[m-1]] + dfx.iloc[i][j][warp_b[m-1]]
                    interpolated_value /= 2
                    m += 1
                
                new_timeseries.append(interpolated_value)
            if len(new_timeseries) != len(dfx.iloc[0][0]):
                new_timeseries.append(new_timeseries[-1])
            body_parts.append(pd.Series(new_timeseries))
        aug_df.append(body_parts)
    aug_df = pd.DataFrame(aug_df)
    aug_df.columns = dfx.columns
    return aug_df