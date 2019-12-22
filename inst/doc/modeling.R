## ----environment, echo = FALSE, message = FALSE, warning=FALSE----------------
knitr::opts_chunk$set(collapse = TRUE, comment = "")
options(tibble.print_min = 6L, tibble.print_max = 6L, width = 80)

library(alookr)
library(ranger)
library(randomForest)

## ----load_data----------------------------------------------------------------
library(mlbench)
data(BreastCancer)

# class of each variables
sapply(BreastCancer, function(x) class(x)[1])

## ----imputate_data, message=FALSE, warning=FALSE------------------------------
library(dlookr)
library(dplyr)

# variable that have a missing value
diagnose(BreastCancer) %>%
  filter(missing_count > 0)

# imputation of missing value
breastCancer <- BreastCancer %>%
  mutate(Bare.nuclei = imputate_na(BreastCancer, Bare.nuclei, Class,
                         method = "mice", no_attrs = TRUE, print_flag = FALSE))

## ----split_data, warning=FALSE------------------------------------------------
library(alookr)

# split the data into a train set and a test set by default arguments
sb <- breastCancer %>%
  split_by(target = Class)

# show the class name
class(sb)

# split the data into a train set and a test set by ratio = 0.6
tmp <- breastCancer %>%
  split_by(Class, ratio = 0.6)

## ----split_summary, warning=FALSE---------------------------------------------
# summary() display the some information
summary(sb)

# summary() display the some information
summary(tmp)

## ----split_check, warning=FALSE-----------------------------------------------
# list of categorical variables in the train set that contain missing levels
nolevel_in_train <- sb %>%
  compare_category() %>% 
  filter(train == 0) %>% 
  select(variable) %>% 
  unique() %>% 
  pull

nolevel_in_train

# if any of the categorical variables in the train set contain a missing level, 
# split them again.
while (length(nolevel_in_train) > 0) {
  sb <- breastCancer %>%
    split_by(Class)

  nolevel_in_train <- sb %>%
    compare_category() %>% 
    filter(train == 0) %>% 
    select(variable) %>% 
    unique() %>% 
    pull
}

## ----show_ratio, warning=FALSE------------------------------------------------
# train set frequency table - imbalanced classes data
table(sb$Class)

# train set relative frequency table - imbalanced classes data
prop.table(table(sb$Class))

# using summary function - imbalanced classes data
summary(sb)

## ----sampling_over, warning=FALSE---------------------------------------------
# to balanced by over sampling
train_over <- sb %>%
  sampling_target(method = "ubOver")

# frequency table 
table(train_over$Class)

## ----sampling_under, warning=FALSE--------------------------------------------
# to balanced by under sampling
train_under <- sb %>%
  sampling_target(method = "ubUnder")

# frequency table 
table(train_under$Class)

## ----sampling_smote, warning=FALSE--------------------------------------------
# to balanced by SMOTE
train_smote <- sb %>%
  sampling_target(seed = 1234L, method = "ubSMOTE")

# frequency table 
table(train_smote$Class)

## ----clean_data, warning=FALSE------------------------------------------------
# clean the training set
train <- train_smote %>%
  cleanse

## ----extract_test, warning=FALSE----------------------------------------------
# extract test set
test <- sb %>%
  extract_set(set = "test")

## ----fit_model, message=FALSE, warning=FALSE----------------------------------
result <- train %>% 
  run_models(target = "Class", positive = "malignant")
result

## ----predict------------------------------------------------------------------
pred <- result %>%
  run_predict(test)
pred

## ----performance1-------------------------------------------------------------
# Calculate performace metrics.
perf <- run_performance(pred)
perf

## ----performance2-------------------------------------------------------------
# Performance by analytics models
performance <- perf$performance
names(performance) <- perf$model_id
performance

## ----performance3-------------------------------------------------------------
# Convert to matrix for compare performace.
sapply(performance, "c")

## ----compare_performance------------------------------------------------------
# Compaire the Performance metrics of each model
comp_perf <- compare_performance(pred)
comp_perf

## ----ROC, fig.height=4, fig.width=7-------------------------------------------
# Plot ROC curve
plot_performance(pred)

## ----cutoff, warning=FALSE, fig.height=4, fig.width=7-------------------------
pred_best <- pred %>% 
  filter(model_id == comp_perf$recommend_model) %>% 
  select(predicted) %>% 
  pull %>% 
  .[[1]] %>% 
  attr("pred_prob")

cutoff <- plot_cutoff(pred_best, test$Class, "malignant", type = "mcc")
cutoff

cutoff2 <- plot_cutoff(pred_best, test$Class, "malignant", type = "density")
cutoff2

cutoff3 <- plot_cutoff(pred_best, test$Class, "malignant", type = "prob")
cutoff3

## ----predit_cutoff------------------------------------------------------------
comp_perf$recommend_model

# extract predicted probability
idx <- which(pred$model_id == comp_perf$recommend_model)
pred_prob <- attr(pred$predicted[[idx]], "pred_prob")

# or, extract predicted probability using dplyr
pred_prob <- pred %>% 
  filter(model_id == comp_perf$recommend_model) %>% 
  select(predicted) %>% 
  pull %>% 
  "[["(1) %>% 
  attr("pred_prob")

# predicted probability
pred_prob  

# compaire Accuracy
performance_metric(pred_prob, test$Class, "malignant", "Accuracy")
performance_metric(pred_prob, test$Class, "malignant", "Accuracy",
                   cutoff = cutoff)

# compaire Confusion Matrix
performance_metric(pred_prob, test$Class, "malignant", "ConfusionMatrix")
performance_metric(pred_prob, test$Class, "malignant", "ConfusionMatrix", 
                   cutoff = cutoff)

# compaire F1 Score
performance_metric(pred_prob, test$Class, "malignant", "F1_Score")
performance_metric(pred_prob, test$Class,  "malignant", "F1_Score", 
                   cutoff = cutoff)
performance_metric(pred_prob, test$Class,  "malignant", "F1_Score", 
                   cutoff = cutoff2)

## ----predit_data, warning=FALSE-----------------------------------------------
data_pred <- train_under %>% 
  cleanse 

set.seed(1234L)
data_pred <- data_pred %>% 
  nrow %>% 
  seq %>% 
  sample(size = 50) %>% 
  data_pred[., ]

## ----predit_final, warning=FALSE----------------------------------------------
pred_actual <- pred %>%
  filter(model_id == comp_perf$recommend_model) %>% 
  run_predict(data_pred) %>% 
  select(predicted) %>% 
  pull %>% 
  "[["(1) %>% 
  factor()

pred_actual

## ----predit_final2, warning=FALSE---------------------------------------------
pred_actual2 <- pred %>%
  filter(model_id == comp_perf$recommend_model) %>% 
  run_predict(data_pred, cutoff) %>% 
  select(predicted) %>% 
  pull %>% 
  "[["(1) %>% 
  factor()

pred_actual2

sum(pred_actual != pred_actual2)

