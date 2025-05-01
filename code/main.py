#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: Differential Privacy for HPC Systems Logs 
Description: Transforming HPC traces about users' jobs submmited in DP data using Laplace mechanism.
Author: Ana Luisa Veroneze Solorzano
Date: September 10, 2024
Version: 1
License: <MIT>

Dependencies:
    - Python 3.8
    - Pandas
    - Sklearn
    - Numpy
    - PYDP
     
Usage:
 
    python main.py -data <input_file> -neighbors <file_path> -q <query_type> -m <method> -c <column_aggregation> -i <iterations> -out <transformed_dataname>

    - (-input_file): pre-processed trace
    - (-neighbors): directory to store information about neighbor datasets
    - (-m) Methods: (1) global DP, (2) local DP
    - (-acc) Desired accuracy from 0-1 for the query of interest 
    - (-data) Dataset in CSV format
    - (-neighbors) Neighbor databases folder
    - (-i) Iterations for the same epsilon
    - (-q) Query: mean, sum, std, var, max, min, count, median
    - (-lib) Library to calculate DP results: PyDP or diffpriv
    - (-c) Sensitive column(s) to apply the query. More than one column can be used, but should be defined individually. Exp: -c sum_jobs -c mean_runtime
    - (-a) Name of the aggregation column for datasets grouped by one column. Exp: Month, System, StartHour
    - (-out) Output file name - to be stored in the CSV file format  
 
""" 

import os, time, argparse, psutil 
import pandas as pd  
from dp_code import *
from localdp import *
from globaldp import *   
import matplotlib.pyplot as plt
import seaborn as sns

CORES_NUM = 8
 
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss
    return mem_info
    
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("\nSYS USAGE:\n{}: Memory consumed: {:,}".format(func.__name__,mem_before, mem_after, mem_after - mem_before))
        return result
    return wrapper

def plot_accuracy_privacy(df, figname):

    df = df.copy()
    df['distance'] = df['Error'].abs()
    grouped = df.groupby('Epsilon').agg(SME=('Error', 'mean'), EUD=('EuD', 'mean')).reset_index()

    # Normalization
    min_sme, max_sme = grouped['SME'].min(), grouped['SME'].max()
    min_eud, max_eud = grouped['EUD'].min(), grouped['EUD'].max()

    grouped['Accuracy'] = 1 - (grouped['SME'] - min_sme) / (max_sme - min_sme)
    grouped['Privacy'] = (grouped['EUD'] - min_eud) / (max_eud - min_eud)
    
    # Find the "sweet-spot" - where we achieve higher accuracy and higher privacy
    best_idx = (grouped['Accuracy'] - grouped['Privacy']).abs().idxmin() 
    best_epsilon = grouped.loc[best_idx, 'Epsilon']
    best_accuracy = grouped.loc[best_idx, 'Accuracy']
    best_privacy = grouped.loc[best_idx, 'Privacy']

    # Plot
    plt.figure(figsize=(10 / 2.54, 8 / 2.54)) 

    sns.lineplot(data=grouped, x='Epsilon', y='Accuracy', label='Accuracy', color='darkblue', linewidth=0.8)
    sns.lineplot(data=grouped, x='Epsilon', y='Privacy', label='Privacy', color='darkgreen', linewidth=0.8)

    plt.scatter(grouped['Epsilon'], grouped['Accuracy'], color='black', s=6, facecolors='none')
    plt.scatter(grouped['Epsilon'], grouped['Privacy'], color='black', s=6, facecolors='none')

    plt.axvline(x=best_epsilon, color='red', linewidth=0.5)
    plt.title(f"Trade-off (ε): {best_epsilon:.3f}\nAccuracy: {best_accuracy:.3f}\nPrivacy: {best_privacy:.3f}", fontsize=8, color='red', pad=10)

    plt.xlabel("Privacy budget (ε)\n(lower ε indicates higher noise)", fontsize=9)
    plt.ylabel("Normalized values", fontsize=9)

    plt.ylim(0, 1)
    plt.xlim(grouped['Epsilon'].min(), grouped['Epsilon'].max())
    plt.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    print(f"Saved figure to {figname}")

def plot_query_result(df, figname):

    # Compute distances for privacy and accuracy
    df = df.copy()
    df['distance'] = df['Error'].abs()

    # Calculate accuracy and privacy for each epsilon
    grouped = df.groupby('Epsilon').agg(SME=('distance', 'mean'), EUD=('EuD', 'mean')).reset_index()

    grouped['min_sme'] = grouped['SME'].min()
    grouped['max_sme'] = grouped['SME'].max()
    grouped['min_eud'] = grouped['EUD'].min()
    grouped['max_eud'] = grouped['EUD'].max()

    grouped['Accuracy'] = (grouped['SME'] - grouped['min_sme']) / (grouped['max_sme'] - grouped['min_sme'])
    grouped['Privacy'] = (grouped['EUD'] - grouped['min_eud']) / (grouped['max_eud'] - grouped['min_eud'])

    # Generate plot - save in the same output directory
    da = df['DA'].iloc[0]
    query = df['Query'].iloc[0]
    col = df['Col'].iloc[0] 
    plt.figure(figsize=(10 / 2.54, 8 / 2.54))

    sns.scatterplot(data=df, x='Epsilon', y='DP_DA', color='black', edgecolor='white', s=10) 
    plt.axhline(y=da, color='red', linewidth=0.3)

    plt.title(f"Real result = {round(da, 2)}", fontsize=8, color='red')
    plt.ylabel(f"Query ({query}): {col}")
    plt.xlabel("Privacy budget (ε)\n(lower ε indicates higher noise)", fontsize=10)

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    print(f"Saved figure to {figname}")

@profile
def main():
    
    parser = argparse.ArgumentParser(description="DP For HPC System Logs:")
    parser.add_argument('-m', type=int, help='Method: [1] Global DP [2] Flipping [3] Laplace.', default=1)
    parser.add_argument('-acc', type=float, help='Desired accuracy from 0-1')

    parser.add_argument('-epsilon', type=float, nargs='+', help='List of epsilon values', default=[0.01])
    parser.add_argument('-data', type=str, help='Dataset in CSV format', default="")
    parser.add_argument('-neighbors', type=str, help='Neighbor databases folder', default="")
    parser.add_argument('-i', type=int, help='Iterations for epsilon', default=250)
    parser.add_argument('-out', type=str, help='Output file name', default="")
    parser.add_argument('-lib', type=str, help='Library: PyDP or diffpriv', default="PYDP")
    parser.add_argument('-c', type=str, action='append', help='Column/Attribute(s) name(s)', default=[])
    parser.add_argument('-agg', type=str, help='Name of the aggregation column/attribute', default="")
    parser.add_argument('-q', type=str, help='Query: Mean, Sum, Variance, Standard Deviation, Median, Count, Max, Min', default="MEAN")

    args = parser.parse_args()    

    epsilon = args.epsilon

    if(args.data):

        df = pd.read_csv(args.data, engine="pyarrow")
        print("\nORIGINAL DATASET --------------------------------------------------------\n")
        print(df) 

    # Global DP: compute DP statistics for different privacy budgets      
    if(args.m == 1):

        if(args.neighbors):

            df = globaldp(args, df, epsilon, CORES_NUM) 
            df.to_csv(args.out, index=False) 
            plot_query_result(df, os.path.splitext(args.out)[0] + '.png')
            plot_accuracy_privacy(df, os.path.splitext(args.out)[0] + '_privacy_acc.png')

        else:

            print("Use code ./preprocessing/main.py to generate neighboor datasets before starting.")
            
    
    # Local DP: add DP perturbation to all entries in the dataset
    elif(args.m == 2):

        random.seed(42)
        INPUT_FILE=args.data 

        # Read dataset  
        df = pd.read_csv(INPUT_FILE, engine="pyarrow")

        # Split the original dataframe into sub-dataframes, one for each column
        # that we want to flip
        # sub_dfs = [(col, df[[col]]) for col in args.c]
        
        # # Use multiprocessing.Pool to apply flipping_prob_noise to each sub-dataframe in parallel
        # with Pool() as pool:
        #     results = pool.map(flipping_prob_noise_parallel, sub_dfs)
        # # Combine the results into a single dataframe
        # df_results = pd.concat(results, axis=1)

        # Apply flipping
        df_results = flipping_prob_noise(df, args.c)

        # Find the equivalent epsilon used by our tool
        # epsilon = find_epsilon(df_results, args.c)

        # Create new column with noise added with Laplace for the same epsilon
        # Also can collect laplace values
        # df_laplace = laplace_noise(df, args.c, epsilon)

        # print(df_results)
        df_results.to_csv(args.out, index=False) 

    # Local DP: with Laplace noise
    elif(args.m == 3):
  
        random.seed(42)
        df = laplace_noise(df, args.c, epsilon[0]) 
        # Write all to the CSV file
        df.to_csv(args.out, index=False)

    else:
        print("Pick a method (-m): (1) global dp, (2) local dp - flipping, (3) local dp - Laplace")

if __name__ == '__main__':

    start_time_exec = time.time()
    main()
    print("Execution time: %s sec" % (time.time() - start_time_exec))