options(crayon.enabled=FALSE)
library(tidyverse) 
library(dplyr)

# Get information about OLCF system logs in a common folder
folder_path <- "./data/OLCF"
csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)

total_users = 0
total_jobs = 0
total_jobs_na = 0
total_jobs_nona = 0
for (csv_file in csv_files) {

    df <- read_csv(csv_file, show_col_types = FALSE)

    cat("Filename:", csv_file, "\n")

    df %>% group_by(user_id) %>% summarise(n=n()) -> df_unique_users
    cat("Number of distinct users:", nrow(df_unique_users), "\n")

    notna_count <- sum(!is.na(df$user_id))
    cat("Jobs by known users:", notna_count, "\n")

    na_count <- sum(is.na(df$user_id))
    cat("Jobs by NA users:", na_count, "\n")

    num_rows <- nrow(df)
    cat("Number of rows:", num_rows, "\n")

    total_users = total_users + nrow(df_unique_users)
    if(na_count != 0){
        total_users = total_users - 1
    }

    total_jobs = total_jobs + nrow(df)
    total_jobs_nona = total_jobs_nona + notna_count
    total_jobs_na = total_jobs_na + na_count
}

cat("\nTotal users - no NA:", total_users, "\n")
cat("Total jobs - no NA users:", total_jobs_nona, "\n")
cat("Total jobs - NA users:", total_jobs_na, " - ", (total_jobs_na*100)/total_jobs, "%\n")
