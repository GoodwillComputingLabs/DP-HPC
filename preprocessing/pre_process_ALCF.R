options(crayon.enabled=FALSE)
library(tidyverse) 
library(dplyr)

# Pre-process data, get most relevant columns

# Set filenames (input and output):
# filename="./data/ALCF/Theta/ANL-ALCF-DJC-THETA_20230101_20231130.csv"
# write_csv(df, "./data/ALCF/Theta/2023_job_trace.csv")

df <- read_csv(filename) %>%
    rename(UserID = USERNAME_GENID,
        ProjectID = PROJECT_NAME_GENID,
        QueueName = QUEUE_NAME,
        Runtime = RUNTIME_SECONDS,
        EligibleQueueTime = QUEUED_WAIT_SECONDS,
        WallTimeRequested = WALLTIME_SECONDS,
        QueuedTimestamp = QUEUED_TIMESTAMP,
        `#NodesRequested` = NODES_REQUESTED,
        `#CoresRequested` = CORES_REQUESTED,
        `#NodeSecondsUsed` = NODES_USED,
        `#CoreSecondsUsed` = CORES_USED,
        StartTimestamp = START_TIMESTAMP,
        EndTimestamp = END_TIMESTAMP) %>%
        select(UserID, ProjectID, QueueName, `#NodesRequested`, `#CoresRequested`,
        WallTimeRequested, QueuedTimestamp, StartTimestamp, EndTimestamp,
        EligibleQueueTime, Runtime, `#NodeSecondsUsed`, `#CoreSecondsUsed`) 



