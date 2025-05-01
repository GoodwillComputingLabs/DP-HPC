import random, math
import datetime as dt
from datetime import datetime
from scipy.stats import norm
from pydp.algorithms.laplacian import BoundedMean
from multiprocessing import Process, Queue, Pool
import numpy as np
import pandas as pd
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from multiprocessing import Process

random.seed(42)

def process_value(args):

    col, sub_df = args

    for value in sub_df[col]:
        print(value)
        
    if new_value in sensitive_values_set.values:
        distances = df_all[df_all['sens_val'] == new_value].reset_index(drop=True)
        flip_prob = random.uniform(distances['dist_cumm'].min(), distances['dist_cumm'].max())
        new_value_index = distances['dist_cumm'].sub(flip_prob).abs().idxmin()
        new_value = distances['max_rss'].values[new_value_index]

    return new_value

#########################################
# ADD LAPLACE NOISE
#########################################
def laplace_noise(df, columns, epsilon):

    new_values = []

    for col in columns:

        datapoints = len(df[col])
        mean_col = df[col].mean()

        noises = np.random.laplace(loc=0, scale=(df[col].max() - df[col].min())/epsilon, size=datapoints) 

        df[col + "LA"] = df[col] + noises 
        # for value in df[col]:
        #     noise = np.random.laplace(loc=0, scale= mean_col/epsilon, size=1) 
        #     # 
        #     # print(noise)
        # #     noise = np.random.laplace(scale=1/epsilon, size=datapoints) 
        #     new_values.append(value + noise)

        # Generate Laplace noise for the entire column in one go
        # noises = np.random.laplace(loc=mean_col, scale=epsilon * mean_col, size=datapoints)
        
        # Create a new column with the noisy data
        # df[col + "LA"] = new_values

        print("Mean Original:", df[col].mean())
        print("Mean with Laplace:", df[col + "LA"].mean())
     
    return df

#########################################
# FIND EPSILON - FLIPPING PROBABILITY
#########################################
def find_epsilon(df, columns):

    for col in columns:

        original_mean = df[col].mean()
        flipped_mean = df[col+"Flip"].mean()
        print("Mean of original:", original_mean)
        print("Mean of noised:", flipped_mean)
        final_epsilon = 0
        final_dp_mean = 0
        final_dist = float('inf')
        for epsilon in np.arange(0.01, 1.5, 0.01):

            x = BoundedMean(epsilon=epsilon, lower_bound=min(df[col]), upper_bound=max(df[col]), dtype="float")
            dp_mean = x.quick_result(list(df[col]))

            # Check if new mean distance from profile noised mean is smaller
            dist = abs(dp_mean - flipped_mean)
            if(dist < final_dist):
                final_dist = dist 
                final_dp_mean = dp_mean
                final_epsilon = epsilon

        print("\n(Laplace Global-DP) Epsilon:", round(final_epsilon, 2), 
            "Mean:", final_dp_mean, "Distance:", round(final_dist, 4))

    return final_epsilon

#########################################
# FLIPPING PROBABILITY
#########################################
def flipping_prob_noise_parallel(args):
    col, sub_df = args
    # apply the original flipping_prob_noise function to the sub-dataframe
    return flipping_prob_noise(sub_df, [col])

# main function here
def flipping_prob_noise(df, columns):
    
    col = columns[0]
    elements = len(df)

    print(df)

    # 1. Get the occurrences of each value
    df_occ = pd.DataFrame(df.groupby(by=col).size().reset_index(name='occ'))
    print("Occurences:\n", df_occ)

    # CLASSIFICATION ################################
    # df_sensitive = df_occ[df_occ['occ'] <= 3] # FOR 5%
    # df_sensitive = df_occ[df_occ['occ'] <= 6] # FOR 25%
    # df_sensitive = df_occ[df_occ['occ'] <= 10] # FOR 50%
    # df_sensitive = df_occ[df_occ['occ'] <= 28] # FOR 90%

    # REGRESSION ################################
    df_sensitive = df_occ[df_occ['occ'] <= 1] # FOR 5%
    #df_sensitive = df_occ[df_occ['occ'] <= 1000] # FOR 25%
    print("Sensitive values:\n", df_sensitive[[col, 'occ']], "\n\n")
    
    sen_rows = (df[col].isin(df_sensitive[col])).sum()
    print("Sensitive rows to be flipped:", sen_rows)

    # 3. Calculate the distances between sensitive values and others
    print("\nCalculating distances...")

    df_all = pd.DataFrame() 
    df_list = []
    for value in df_sensitive[col]:
        
        df_temp = df_occ.copy()

        df_temp['sens_val'] = value
        df_temp['dist'] = abs(df_temp[col] - value)
        df_temp['dist_dif'] = df_temp['dist'].sum() - df_temp['dist']
        df_temp['dist_cumm'] = df_temp['dist_dif'].cumsum()

        # for v in range(len(df_temp['dist_dif'].tolist())):
        #     a.append(sum(df_temp['dist_dif'].tolist()[0:v]))
        # df_temp['dist_cumm'] = a

        # Append the resulting DataFrame to the list
        df_list.append(df_temp)
    
    df_all = pd.concat(df_list, axis=0, ignore_index=True)
    print(df_all)

    # 4. For each sensitive value in the dataset, calculate to which value to flip
    new_col = []

    print(set(df_sensitive['max_rss']))
    sub_dfs = [(col, df_all[[col]]) for col in ['max_rss']]
    with Pool(2) as pool:
        # results = pool.map(flipping_prob_noise_parallel, sub_dfs)
        # Process values in parallel
        args = zip(df[col], set(df_sensitive['max_rss']), sub_dfs)
        # new_col = list(pool.map(process_value, args))

    # Combine the results into a single dataframe
    # df_results = pd.concat(results, axis=1)

    # print("Flipping values...")
    # new_col = []
    # sensitive_values_set = set(df_sensitive[col])
    # for value in df[col]:

    #     new_value = value
    #     # if(value in df_sensitive[col].values):
    #     if(value in sensitive_values_set):
    #     # if(df_sensitive[col].isin([value]).any()):
    #         distances = df_all[df_all['sens_val'] == new_value].reset_index(drop=True)
    #         # Get a probability to flip
    #         flip_prob = random.uniform(distances['dist_cumm'].min(), distances['dist_cumm'].max())
    #         # Find the nearest value to this probability 
    #         new_value_index = distances['dist_cumm'].sub(flip_prob).abs().idxmin()
    #         new_value = distances[col].values[new_value_index]
    #         # print("Old - New:", value, new_value)

    #     new_col.append(new_value)

    df[col+'_flip'] = new_col
    print(df[[col, col+'_flip']])
    
    return df