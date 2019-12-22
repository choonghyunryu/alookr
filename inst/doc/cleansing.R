## ----environment, echo = FALSE, message = FALSE, warning=FALSE----------------
knitr::opts_chunk$set(collapse = TRUE, comment = "")
options(tibble.print_min = 4L, tibble.print_max = 4L, width = 80)

library(alookr)

## ----create_data--------------------------------------------------------------
# create sample dataset
set.seed(123L)
id <- sapply(1:1000, function(x)
  paste(c(sample(letters, 5), x), collapse = ""))

year <- "2018"

set.seed(123L)
count <- sample(1:10, size = 1000, replace = TRUE)

set.seed(123L)
alpha <- sample(letters, size = 1000, replace = TRUE)

set.seed(123L)
flag <- sample(c("Y", "N"), size = 1000, prob = c(0.1, 0.9), replace = TRUE)

data_exam <- data.frame(id, year, count, alpha, flag, stringsAsFactors = FALSE)

# structure of dataset
str(data_exam)

# summary of dataset
summary(data_exam)

## ----cleanse------------------------------------------------------------------
# cleansing dataset
newDat <- cleanse(data_exam)

# structure of cleansing dataset
str(newDat)

## ----cleanse_2----------------------------------------------------------------
# cleansing dataset
newDat <- cleanse(data_exam, uniq_thres = 0.03)

# structure of cleansing dataset
str(newDat)

## ----cleanse_3----------------------------------------------------------------
# cleansing dataset
newDat <- cleanse(data_exam, uniq = FALSE)

# structure of cleansing dataset
str(newDat)

## ----cleanse_4----------------------------------------------------------------
# cleansing dataset
newDat <- cleanse(data_exam, char = FALSE)

# structure of cleansing dataset
str(newDat)

## ----cleanse_5----------------------------------------------------------------
data_exam$flag[1] <- NA 

# cleansing dataset
newDat <- cleanse(data_exam, missing = TRUE)

# structure of cleansing dataset
str(newDat)

## ----treatment_corr-----------------------------------------------------------
# numerical variable
x1 <- 1:100
set.seed(12L)
x2 <- sample(1:3, size = 100, replace = TRUE) * x1 + rnorm(1)
set.seed(1234L)
x3 <- sample(1:2, size = 100, replace = TRUE) * x1 + rnorm(1)

# categorical variable
x4 <- factor(rep(letters[1:20], time = 5))
set.seed(100L)
x5 <- factor(rep(letters[1:20 + sample(1:6, size = 20, replace = TRUE)], time = 5))
set.seed(200L)
x6 <- factor(rep(letters[1:20 + sample(1:3, size = 20, replace = TRUE)], time = 5))
set.seed(300L)
x7 <- factor(sample(letters[1:5], size = 100, replace = TRUE))

exam <- data.frame(x1, x2, x3, x4, x5, x6, x7)
str(exam)
head(exam)

# default case
exam_01 <- treatment_corr(exam)
head(exam_01)

# not removing variables
treatment_corr(exam, treat = FALSE)

# Set a threshold to detecting variables when correlation greater then 0.9
treatment_corr(exam, corr_thres = 0.9, treat = FALSE)

# not verbose mode
exam_02 <- treatment_corr(exam, verbose = FALSE)
head(exam_02)

