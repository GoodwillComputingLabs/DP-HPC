#  Privacy-Preserving Transformations of HPC Traces using Differential Privacy

Differential privacy (DP) provides mathematical guarantees that data can remain private even after transformations to hide Personal Identifiable Information (PII). Our toolset aims to facilitate using DP to protect sensitive data in HPC system logs for Global DP and local. In Global DP the data owner knows the queries the analyst will perform over in advance. The data owner will then release the DP query results with a privacy budget guarantee. In Local DP, noise is added to each individual data point before the dataset values are aggregated, the data owner usually prioritizes higher privacy protection by adding more noise since any process applied to the data is unpredictable.

<img src="./img/global-local-dp.png" width="500">

## Configuring environment

We used Python version 3.8.10 and R version 4.2.3, run the following commands to install the required libraries:

```
conda create -n myenv python=3.8.10
conda activate myenv
pip install -r requirements.txt
```

```
install.packages(c("tidyverse", "data.table", "ggforce", "arrow", "hms", "scales", "patchwork", "Hmisc"))
``` 

Installation of ggpatern and tidyverse migh be needed for R packages:
```sh
sudo apt install libgdal-dev
sudo apt-get install libharfbuzz-dev
```

## Open HPC traces: 

- OLCF: https://doi.ccs.ornl.gov/ui/doi/334 
- ALCF - all DIM_JOB_COMPOSITE files: https://reports.alcf.anl.gov/data 

We recommend downloading the data and putting them in the following directories, although our toolset allow you to define all paths via command line:
```sh
mkdir -p data/ALCF
mkdir -p data/OLCF
```

## Data processing

### Parsing 

Parse data to get the most relevant columns:

```sh
Rscript preprocess_data/pre_process_ALCF.R
Rscript preprocess_data/pre_process_OLCF.R
```

Example:
```
UserID,ProjectID,QueueName,#NodesRequested,#CoresRequested,WallTimeRequested,QueuedTimestamp,StartTimestamp,EndTimestamp,EligibleQueueTime,Runtime,#NodeSecondsUsed,#CoreSecondsUsed
67550677511631,40773989651712,backfill,1024,16384,1500,2014-12-31 09:22:28.000000,2014-12-31 23:58:30.000000,2015-01-01 00:19:27.000000,52563,1257,1287168,20594688
61648062878124,59933165581865,prod-short,4096,65536,21600,2014-12-18 17:39:38.000000,2014-12-31 18:19:20.000000,2015-01-01 00:19:57.000000,112830,21637,88625152,1418002432
13158054236861,59813733799296,prod-short,512,8192,7200,2014-12-30 16:03:16.000000,2014-12-31 23:26:28.000000,2015-01-01 00:27:14.000000,112992,3646,1866752,29868032 
```

### Pre-processing

Create neighbors and aggrega data to apply transformations:

```sh 
  python preprocessing/main.py -indir data/ALCF/Cooley/ -outdir data/globaldp/Cooley/ -sysname Cooley -col UserID
  python preprocessing/main.py -indir data/ALCF/Theta/ -outdir data/globaldp/Theta/ -sysname Theta -col UserID
  python preprocessing/main.py -indir data/ALCF/Mira/ -outdir data/globaldp/Mira/ -sysname Mira -col UserID
  python preprocessing/main.py -indir data/OLCF/ -outdir data/globaldp/Titan/ -sysname Titan -col user_id
  # Optional, also aggregate by multiple columns
  python preprocessing/main.py -indir data/ALCF/Cooley/ -outdir data/CooleyMultiple/ -cols "UserID" -cols "WeekDay"
```

## Execution

Toolset parameters:

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
