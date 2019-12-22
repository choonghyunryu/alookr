## ----environment, echo = FALSE, message = FALSE, warning=FALSE----------------
knitr::opts_chunk$set(collapse = TRUE, comment = "")
options(tibble.print_min = 6L, tibble.print_max = 6L, width = 80)

library(alookr)

## ----create_data--------------------------------------------------------------
# Credit Card Default Data
head(ISLR::Default)

# structure of dataset
str(ISLR::Default)

# summary of dataset
summary(ISLR::Default)

## ----splits, message=FALSE----------------------------------------------------
library(alookr)
library(dplyr)

# Generate data for the example
sb <- ISLR::Default %>%
  split_by(default, seed = 6534)

sb

## ----attr---------------------------------------------------------------------
sb_attr <- attributes(sb)

# The third attribute, row.names, is a vector that is very long and excluded from the output.
sb_attr[-3]

## ----summ---------------------------------------------------------------------
summary(sb)

## ----compare_category---------------------------------------------------------
sb %>%
  compare_category()

# compare variables that are character data types.
sb %>%
  compare_category(add_character = TRUE)

# display marginal
sb %>%
  compare_category(margin = TRUE)

# student variable only
sb %>%
  compare_category(student)

sb %>%
  compare_category(student, margin = TRUE)

## ----compare_numeric----------------------------------------------------------
sb %>%
  compare_numeric()

# balance variable only
sb %>%
  compare_numeric(balance)

## ----compare_plot, fig.height=5, fig.width=6, message=FALSE-------------------
# income variable only
sb %>%
  compare_plot("income")

# all varibales
sb %>%
  compare_plot()

## ----create_dataset-----------------------------------------------------------
defaults <- ISLR::Default
defaults$id <- seq(NROW(defaults))

set.seed(1)
defaults[sample(seq(NROW(defaults)), 3), "student"] <- NA
set.seed(2)
defaults[sample(seq(NROW(defaults)), 10), "balance"] <- NA

sb_2 <- defaults %>%
  split_by(default)

sb_2 %>%
  compare_diag()

sb_2 %>%
  compare_diag(add_character = TRUE)

sb_2 %>%
  compare_diag(uniq_thres = 0.0005)

## ----extract_set--------------------------------------------------------------
train <- sb %>%
  extract_set(set = "train")

test <- sb %>%
  extract_set(set = "test")

dim(train)

dim(test)

## ----sampling_target----------------------------------------------------------
# under-sampling with random seed
under <- sb %>%
  sampling_target(seed = 1234L)

under %>%
  count(default)

# under-sampling with random seed, and minority class frequency is 40%
under40 <- sb %>%
  sampling_target(seed = 1234L, perc = 40)

under40 %>%
  count(default)

# over-sampling with random seed
over <- sb %>%
  sampling_target(method = "ubOver", seed = 1234L)

over %>%
  count(default)

# over-sampling with random seed, and k = 10
over10 <- sb %>%
  sampling_target(method = "ubOver", seed = 1234L, k = 10)

over10 %>%
  count(default)

# SMOTE with random seed
smote <- sb %>%
  sampling_target(method = "ubSMOTE", seed = 1234L)

smote %>%
  count(default)

# SMOTE with random seed, and perc.under = 250
smote250 <- sb %>%
  sampling_target(method = "ubSMOTE", seed = 1234L, perc.under = 250)

smote250 %>%
  count(default)

