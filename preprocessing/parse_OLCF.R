options(crayon.enabled=FALSE)
library(tidyverse) 
library(dplyr)

# Pre-process original downloaded data, rename some columns and get the most relevant 
Set filenames (input and output):
filename="./data/OLCF/.csv"
write_csv(df, "./data/OLCF/Titan/.csv")

df <- read_csv(filename) %>%
    rename(user_id = USERNAME_GENID,
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
        select(user_id, ProjectID, QueueName, `#NodesRequested`, `#CoresRequested`,
        WallTimeRequested, QueuedTimestamp, StartTimestamp, EndTimestamp,
        EligibleQueueTime, Runtime, `#NodeSecondsUsed`, `#CoreSecondsUsed`) 