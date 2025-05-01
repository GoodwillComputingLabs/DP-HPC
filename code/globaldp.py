import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, ceil 
from dp_code import *
from localdp import * 
from multiprocessing import  Pool
from itertools import chain

PARALLEL = True

########################################################################
# PLOT FINAL RESULT
########################################################################
def plot_accuracy(df, title=" "):

    fig = plt.figure(figsize = (4, 4))

    labels = [0, 1, 2, 3, r"$\infty$"]
    plt.xticks(list(np.arange(0.01, 1, 0.1)))
    plt.xticks(list(np.arange(0.01, 1, 0.1)))

    plt.plot(df['Epsilon'], df['EuD_norm'], color ='g', markersize=2, label="Privacy")
    plt.plot(df['Epsilon'], df['Accuracy_norm'], color = 'b', markersize=2, label="Accuracy")
    plt.xlabel("Epsilon")
    plt.ylabel("Norm. Accuracy and Privacy")
    plt.legend(loc="upper right")
    plt.title(title, size=10)

    return plt

########################################################################
# QUERIES FUNCTIONS
########################################################################
def get_mean(df_col):
    return np.mean(df_col)

def get_sum(df_col):
    return np.sum(df_col)

def get_std(df_col):
    return np.std(df_col)

def get_var(df_col):
    return np.var(df_col)

def get_max(df_col):
    return np.max(df_col)

def get_min(df_col):
    return np.min(df_col)

def get_count(df_col):
    return np.size(df_col)

def get_median(df_col):
    return np.median(df_col)

########################################################################
# SENSITIVITY FUNCTIONS
########################################################################
def get_mean_sensitivity(df_col, min_value, max_value):
    return (max_value - min_value) / len(df_col)

def get_sum_sensitivity(df_col, min_value, max_value):
    return (max_value - min_value)

def get_std_sensitivity(df_col, min_value, max_value):
    print("Calculating sensitivity")
    print(np.subtract.outer(df_col, df_col))
    return np.sqrt(np.max(np.square(np.subtract.outer(df_col, df_col)))) 

def get_var_sensitivity(df_col, min_value, max_value):
    return np.max(np.square(np.subtract.outer(df_col, df_col)))

def get_max_sensitivity(df_col, min_value, max_value):
    return max_value

def get_min_sensitivity(df_col, min_value, max_value):
    return min_value

def get_count_sensitivity(df_col, min_value, max_value):
    return 1

def get_median_sensitivity(df_col, min_value, max_value):

    sorted_values = sorted(df_col)
    n = len(sorted_values)
    if n % 2 == 0:
        sensitivity = (sorted_values[n//2] - sorted_values[(n//2) - 1])/2
    else:
        sensitivity = 0

    return sensitivity

########################################################################
# AUXILIARY FUNCTIONS
########################################################################
def min_max_normalization(df):

    # g = df.groupby('Column').Accuracy
    # df['Accuracy_norm'] = (df.Accuracy - g.transform('min'))/(g.transform('max') - g.transform('min'))

    g = df.groupby('Column').EuD
    df['EuD_norm'] = (df.EuD - g.transform('min'))/(g.transform('max') - g.transform('min'))

    return df   

def calculate_dbs(batch, query, epsilon, agg, i, q, lib, val, col, query_df, query_dp):

    tmp_eud = 0
    all_data = []

    for filename in batch:

        # Read DB
        df_neighbor = pd.read_csv(filename, engine="pyarrow", usecols=[col])
        # Get original and DP query
        query_df_neighbor = query(df_neighbor[col].astype(float))
        query_dp_neighbor = dp_global(lib, i, epsilon, df_neighbor[col], q, low=min(df_neighbor[col]), up=max(df_neighbor[col]))

        tmp_eud += ((query_df - query_dp) ** 2) + ((query_df_neighbor - query_dp_neighbor) ** 2)

        # Put all results in a list
        all_data.append((query_df_neighbor, query_dp_neighbor))

    return all_data, tmp_eud

def globaldp(args, df, EPSILON, CORES_NUM):
        
    print("\n- Columns selected:", args.c)

    # Statistical Methods
    if(args.q.upper() == "MEAN"):
        query = get_mean
        sensitivity = get_mean_sensitivity
    elif(args.q.upper() == "SUM"):
        query = get_sum
        sensitivity = get_sum_sensitivity
    elif(args.q.upper() == "STD"):
        query = get_std
        sensitivity = get_std_sensitivity
    elif(args.q.upper() == "VAR"):
        query = get_var
        sensitivity = get_var_sensitivity
    elif(args.q.upper() == "MAX"):
        query = get_max
        sensitivity = get_max_sensitivity
    elif(args.q.upper() == "MIN"):
        query = get_min
        sensitivity = get_min_sensitivity
    elif(args.q.upper() == "COUNT"):
        query = get_count
        sensitivity = get_count_sensitivity
    elif(args.q.upper() == "MEDIAN"):
        query = get_median
        sensitivity = get_median_sensitivity
    else:
        return

    print("- Query:", args.q.upper())
    
    # Dataframe to store final values and save as csv
    df_all = pd.DataFrame()

    # If aggregation defined get the unique values for it
    if(args.agg):
        unique_val = df[args.agg].unique()
        df_original = pd.DataFrame()
        df_original = df.copy()
        print("- Dataset aggregation:", unique_val)
    else:
        unique_val = " "

    BATCH_SIZE = ceil(len(os.listdir(args.neighbors)) / CORES_NUM)
    print("- Batch size:", BATCH_SIZE)
    
    # For the distinct values in the aggregation column (if defined)
    for val in unique_val:

        # Dataframe to store results for each attribute
        df_local = pd.DataFrame()
        
        # Filter dataframe to get only entries for the current query
        if(args.agg):
            df = df_original[df_original[args.agg] == val]

        # For each unique column value 
        len_cols = len(list(args.c))
        col_names = list(args.c)
        for col in col_names:

            print("- Aggregation column:", val)
            print("- Column to query:", col)
            # df[col] = df[col].round(0)

            # GET ORIGINAL QUERY RESULT:
            min_col = min(df[col])
            max_col = max(df[col])
            query_df = query(df[col])
            query_sensitivity = round(sensitivity(df[col],min_col,max_col),2)
            print("- Original query result:", query_df)
            print("- Sensitivity:" + str(query_sensitivity) + "\n")

            library = args.lib.upper()
            all_files = [os.path.join(args.neighbors, file) for file in os.listdir(args.neighbors)]
            batches = [all_files[i:i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]

            # For all epsilons defined
            for e in EPSILON:
                
                lower_bound = -(query_sensitivity / e)
                upper_bound = (query_sensitivity / e)

                e = round(e, 3)
                all_data = []
                scale_e = e/len_cols

                data_to_append = []
                # Repetition of the same epsilon:
                for rep in range(0, args.i):
                    
                    # GET ORIGINAL DATASET DP QUERY
                    query_dp = dp_global(library, args.i, scale_e, df[col], args.q.upper(), low=lower_bound, up=upper_bound)                      

                    if(PARALLEL):

                        # Divide all DB files for each process with sizes defined by the user in the header
                        p = Pool(processes=CORES_NUM)
                        # Apply DP for each neighbor file in parallel
                        results = p.starmap(calculate_dbs, [(batch, query, scale_e, args.agg, args.i, args.q.upper(), args.lib.upper(), val, col, query_df, query_dp) for batch in batches])
                        p.close()
                        p.join()

                        all_data, tmp_eud = zip(*results)
                        df_tmp = pd.DataFrame(list(chain.from_iterable(all_data)), columns = ['DB', 'DP_DB'])

                    else:
                        all_data, tmp_eud = calculate_dbs(os.scandir(args.neighbors), query, scale_e, args.agg, args.i, args.q.upper(), args.lib.upper(), val, col, query_df, query_dp)
                        df_tmp = pd.DataFrame(tuple(all_data), columns = ['DB', 'DP_DB'])

                    # Create dataframe with values from all columns
                    df_tmp['aux'] = 1
                    df_tmp = df_tmp.groupby(['aux'], as_index=False)[['DB', 'DP_DB']].mean()
                    df_tmp['DP_DA'] = query_dp
                    df_tmp['Rep'] = rep
                    df_tmp['DA'] = query_df
                    df_tmp['Epsilon'] = e
                    df_tmp['Col'] = col
                    df_tmp['Error'] = (query_df - query_dp)
                    df_tmp['EuD'] = sqrt(np.sum(tmp_eud))

                    # Concatenate the results for each column
                    df_local = pd.concat([df_local, df_tmp], axis=0) 

                print("Epsilon:", str(round(e, 3)))
                print("DP result:", round(df_local['DP_DA'].mean(), 2))
                

        # Normalize EuD, acc_DA and acc_DB data using min max normalization
        # df = min_max_normalization(df)
        # if(args.agg):
        #     df_local[args.agg] = val

        df_local['Query'] = args.q
        df_all = pd.concat([df_all, df_local], axis=0)

    return df_all 
    