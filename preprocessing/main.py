#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: Pre-processing data 
Author: Ana Luisa Veroneze Solorzano
Date: September 10, 2024
Version: 1
License: <MIT>

Dependencies:
    - Python 3.8
    - Pandas
     
Usage:
 
    python main.py -indir <input_directory> -outdir <output_directory> -sysname <HPC_system_name> -col <column_name>
    
    - (-indir): Input directory to find the files to aggregate
    - (-infile): 'Input file name for a single file.
    - (-outdir): Output directory for aggregated file - a directory called "neighbors" will be created inside it to store them.
    - (-sysname): Name of the system to create a column in the aggregated data
    - (-col): Column name to aggregate
    - (-cols): If using multiple columns select each by calling this parameter
    
"""

import os, time, argparse 
import pandas as pd  
from pandas.api.types import is_string_dtype 
 
# Aggregate all data (numeric and textual) into the column of interest and create a csv file
def join_dataset(inputfile, col, systemname):

    df = pd.read_csv(inputfile, engine="python")
    df["System"] = systemname

    # Filter numeric columns and aggregate them
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num = df.select_dtypes(include=numerics)

    # If aggregation column is not numeric just add it to df_num to group
    if(is_string_dtype(df[col])):
        df_num[col] = df[col]

    df_sum = df_num.groupby(col).sum().add_suffix('_sum').reset_index(drop=True)
    print(df_sum)
    df_mean = df_num.groupby(col).mean().add_suffix('_mean').reset_index(drop=True)
    print(df_mean)
    df_median = df_num.groupby(col).median().add_suffix('_median').reset_index(drop=True)
    print(df_median)

    # Get number of rows for each distinct value for the col for number of jobs
    df_count = df.groupby([col])[col].count().reset_index(name = "Jobs")

    # Join all calculated data:
    df_all_num = df_count.join(df_sum).join(df_mean)

    return df_all_num

def join_datasets(fp, col):

    # Iterate directory and join all files inside directory
    df = pd.DataFrame()
    for (dir_path, dir_names, file_names) in os.walk(fp):
        if(file_names):
            system = os.path.basename(dir_path)
            for f in file_names:
                print("File name:", os.path.join(dir_path, f))
                df_local = pd.read_csv(os.path.join(dir_path, f), engine="python")
                df_local["System"] = system
                df = pd.concat([df, df_local], ignore_index=True)

    print("ORIGINAL:")
    print(df)
    dfna = df.dropna()
    # df["WeekDay"] = df["QueuedTimestamp"].dt.hour
    print("REMOVING NAs:")
    print(dfna)

    print(df.columns)
    print(df[col])

    # Aggregate all data (numeric and textual) into the column of interest
    # and create a csv file

    # Filter numeric columns and aggregate them
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num = df.select_dtypes(include=numerics)

    # If aggregation column is not numeric just add it to df_num to group
    if(is_string_dtype(df[col])):
        df_num[col] = df[col]

    df_sum = df_num.groupby(col).sum().add_suffix('_sum').reset_index(drop=True)
    print(df_sum)
    df_mean = df_num.groupby(col).mean().add_suffix('_mean').reset_index(drop=True)
    print(df_mean)
    df_median = df_num.groupby(col).median().add_suffix('_median').reset_index(drop=True)
    print(df_median)

    # Get number of rows for each distinct value for the col for number of jobs
    df_count = df.groupby([col])[col].count().reset_index(name = "Jobs")
    print(df_count)

    # Join all numeric data:
    df_all_num = df_count.join(df_sum).join(df_mean)

    # Filter textual columns and store them in a list
    # df_str = df.select_dtypes(include=['object'])
    # print(df_str.columns)
    # df_all = pd.DataFrame(df_all_num)

    # for str_col in df_str.columns:
    #     df_str = df.groupby(col)[str_col].apply(list).reset_index(drop=True)
    #     df_all = df_all.join(df_str)

    # # Join all and return to save in a CSV file
    # print(df_all)
    # return df_all
    return df_all_num

def join_datasets_multiplecol(fp, cols):

    # Iterate directory and join all files
    df = pd.DataFrame()

    for (dir_path, dir_names, file_names) in os.walk(fp):
        if(file_names):
            system = os.path.basename(dir_path)
            for f in file_names:
                print(os.path.join(dir_path, f))
                df_local = pd.read_csv(os.path.join(dir_path, f), engine="pyarrow")
                df_local["System"] = system
                # df_local["start_time"] = pd.to_datetime(df_local["start_time"])
                df_local["WeekDay"] = df_local["QueuedTimestamp"].dt.dayofweek
                # df_local["HourDay"] = df_local["start_time"].dt.hour
                df = pd.concat([df, df_local], ignore_index=True)

    print("ORIGINAL:")
    print(df)
    df = df.dropna()
    print("REMOVING NAs:")
    print(df)
    print("Grouping by:", cols)

    # Filter numeric columns and aggregate them
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num = df.select_dtypes(include=numerics)
    df_sum = df_num.groupby(cols).sum().add_suffix('_sum').reset_index(drop=True)
    df_mean = df_num.groupby(cols).mean().add_suffix('_mean').reset_index(drop=True)

    # Get number of rows for each distinct value for the col for number of jobs
    df_count = df.groupby(cols).size().reset_index(name='Jobs')
    # df_count = df.groupby(cols)['max_rss'].max().reset_index(name='Maxmem')
    # df_count['Maxmem'] = round(df_count['Maxmem']/1000, 2) # memory in MB
    
    print(df_count)

    # Join all numeric data:
    df_all_num = df_count.join(df_sum).join(df_mean)

    # return df_count
    return df_all_num

def create_neighbors(df, outdir):

    # Create a list of DataFrames with each row removed
    dfs_without_rows = [df.drop(i) for i in range(len(df))]

    # Write each DataFrame
    for i, df_without_row in enumerate(dfs_without_rows):
        df_without_row.to_csv(outdir + str(i) + ".csv", index=False)

def main():

    parser = argparse.ArgumentParser(description="Pre-process logs for DP:")
    parser.add_argument('-indir', type=str, help='Input file path.')
    parser.add_argument('-infile', type=str, help='Input file name for a single file.')
    parser.add_argument('-outdir', type=str, help='Original dataset directory.')
    parser.add_argument('-sysname', type=str, help='System name to create new column.')
    parser.add_argument('-col', type=str, help='Unique column name.', default="")
    parser.add_argument('-cols', type=str, action='append', help='Multiple columns names.', default=[]) 
    parser.add_argument('-neighbors_only', type=bool, help='Only create neighbor files.', default=False) 
    args = parser.parse_args()   
    
    # If multiple columns selected
    if(args.cols != []):
        
        df_all = join_datasets_multiplecol(args.indir, args.cols)
        df_all.to_csv(args.outdir + "/all.csv", index=False)

    else:

        output_neighbors = args.outdir + "/neighbors/"
        if not os.path.exists(output_neighbors):
            os.makedirs(output_neighbors)

        # Just create neighbor datafiles
        if(args.neighbors_only):
            df_all = pd.read_csv(args.indir + "/all.csv")
            print("Creating neighbors...")
            create_neighbors(df_all, output_neighbors)
            
        else:
            print("Writting D and D'...")

            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

            # Join all datasets in a folder into one CSV with unique ID column
            if(os.path.exists(args.indir)):
                start_time = time.time()
                df_all = join_datasets(args.indir, args.col)
                df_all.to_csv(args.outdir + "/all.csv", index=False)
            
            df_all = pd.read_csv(args.outdir + "/all.csv")
            print("Execution time to create aggregated data: %s sec" % (time.time() - start_time))

            print("Creating neighbors...") 
            start_time = time.time()
            create_neighbors(df_all, output_neighbors)
            print("Execution time to create neighbors: %s sec" % (time.time() - start_time))

if __name__ == '__main__':

    main()