packages <- c("ggplot2", "dplyr", "tidyr", "readr", "caret", "stats", "rmarkdown", "knitr")
install.packages(setdiff(packages, rownames(installed.packages())), repos = "https://cloud.r-project.org")
