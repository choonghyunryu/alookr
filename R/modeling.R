#=======================================================
# fit individual model
#=======================================================
#' @importFrom stats glm
#' @importFrom stats binomial
classifier_logistic <- function(.data, target, positive) {
  formula <- paste(target, ".", sep = "~")

  model <- glm(formula, family = binomial, data = .data)

  MASS::stepAIC(model, trace = FALSE)
}

classifier_rpart <- function(.data, target, positive) {
  formula <- paste(target, ".", sep = "~")

  rpart::rpart(formula, data = .data)
}

#' @importFrom stats as.formula
classifier_ctree <- function(.data, target, positive) {
  formula <- as.formula(paste(target, ".", sep = "~"))

  party::ctree(formula, data = .data)
}

#' @importFrom stats as.formula
classifier_randomForest <- function(.data, target, positive) {
  formula <- as.formula(paste(target, ".", sep = "~"))

  randomForest::randomForest(formula, data = .data)
  #randomForest::randomForest(.data %>% select(-target), .data %>% select(target) %>% pull)
}

#' @importFrom stats as.formula
classifier_ranger <- function(.data, target, positive) {
  formula <- as.formula(paste(target, ".", sep = "~"))

  ranger::ranger(formula, data = .data, probability = TRUE)
}

#' @importFrom xgboost xgboost
classifier_xgboost <- function(.data, target, positive) {
  train <- .data %>% 
    select(-target) %>% 
    data.matrix
  
  label <- .data %>% 
    select(variable = target) %>% 
    mutate(variable = ifelse(variable == positive, 1, 0)) %>% 
    pull 
  
  xgboost::xgboost(data = train, label = label, eta = 1,
                   nrounds = 3, objective = "binary:logistic", verbose = 0)
}

#=======================================================
# dispatcher fit models
#=======================================================
classifier_dispatch <- function(model = c("logistic", "rpart", "ctree", 
                                          "randomForest", "ranger", "xgboost"),
                                .data, target, positive) {

  model <- paste("classifier", match.arg(model), sep = "_")

  do.call(model, list(.data = .data, target = target, positive = positive))
}


#' Fit binary classification model
#'
#' @description Fit some representative binary classification models.
#' @param .data A train_df. Train data to fit the model. It also supports tbl_df, 
#' tbl, and data.frame objects.
#' @param target character. Name of target variable.
#' @param positive character. Level of positive class of binary classification.
#' @param models character. Algorithm types of model to fit. See details. 
#' default value is c("logistic", "rpart", "ctree", "randomForest", "ranger").
#'
#' @details Supported models are functions supported by the representative 
#' model package used in R environment.
#' The following binary classifications are supported:
#' \itemize{
#' \item "logistic" : logistic regression by glm() in stats package.
#' \item "rpart" : recursive partitioning tree model by rpart() in rpart package.
#' \item "ctree" : conditional inference tree model by ctree() in party package.
#' \item "randomForest" : random forest model by randomForest() in 
#' randomForest package.
#' \item "ranger" : random forest model by ranger() in ranger package.
#' \item "xgboost" : XGBoosting model by xgboost() in xgboost package.
#' }
#' 
#' run_models() executes the process in parallel when fitting the model. 
#' However, it is not supported in MS-Windows operating system and RStudio 
#' environment.
#'
#' @return model_df. results of fitted model.
#' model_df is composed of tbl_df and contains the following variables.:
#' \itemize{
#' \item step : character. The current stage in the model fit process. 
#' The result of calling run_models() is returned as "1.Fitted".
#' \item model_id : character. Type of fit model.
#' \item target : character. Name of target variable.
#' \item is_factor : logical. Indicates whether the target variable is a factor. 
#' \item positive : character. Level of positive class of binary classification.
#' \item negative : character. Level of negative class of binary classification. 
#' \item fitted_model : list. Fitted model object.
#' }
#'
#' @examples
#' library(dplyr)
#'
#' # Divide the train data set and the test data set.
#' sb <- rpart::kyphosis %>%
#'   split_by(Kyphosis)
#'
#' # Extract the train data set from original data set.
#' train <- sb %>%
#'   extract_set(set = "train")
#'
#' # Extract the test data set from original data set.
#' test <- sb %>%
#'   extract_set(set = "test")
#'
#' # Sampling for unbalanced data set using SMOTE(synthetic minority over-sampling technique).
#' train <- sb %>%
#'   sampling_target(seed = 1234L, method = "ubSMOTE")
#'
#' # Cleaning the set.
#' train <- train %>%
#'   cleanse
#'
#' # Run the model fitting.
#' result <- run_models(.data = train, target = "Kyphosis", positive = "present")
#' result
#'
#' # Run the several kinds model fitting by dplyr
#' train %>%
#'   run_models(target = "Kyphosis", positive = "present")
#'
#' # Run the logistic model fitting by dplyr
#' train %>%
#'   run_models(target = "Kyphosis", positive = "present", models = "logistic")
#' @importFrom stats density
#' @export
run_models <- function(.data, target, positive,
                       models = c("logistic", "rpart", "ctree", "randomForest", 
                                  "ranger", "xgboost")) {
  if (dlookr::get_os() == "windows" || .Platform$GUI == "RStudio") {
    future::plan(future::sequential)
  } else {
    future::plan(future::multiprocess)
  }

  actual_target <- pull(.data, target)
  
  flag_factor <- is.factor(actual_target)
  
  level <- if (flag_factor) levels(actual_target) else unique(actual_target)
  
  negative <- setdiff(level, positive)
  
  result <- purrr::map(models, ~future::future(classifier_dispatch(.x, .data, target, positive))) %>%
    tibble::tibble(step = "1.Fitted", model_id = models, target = target, is_factor = flag_factor,
                   positive = positive, negative = negative,  
                   fitted_model = purrr::map(., ~future::value(.x)))

  result <- result[, -1]

  class(result) <- append("model_df", class(result))

  result
}

#=======================================================
# predict individual model
#=======================================================
#' @importFrom stats predict
#' @rawNamespace import(randomForest, except = c(margin, combine, importance))
#' @import ranger
#' @rawNamespace import(xgboost, except = c(slice))
#'
predictor <- function(model, .data, target, positive, negative, is_factor, 
                      cutoff = 0.5) {
  model_class <- is(model)[1]

  pred <- switch(model_class,
                 glm = predict(model, newdata = .data, type = "response"),
                 rpart = predict(model, newdata = .data, 
                                 type = "prob")[, positive],
                 BinaryTree = sapply(predict(model, newdata = .data, 
                                             type = "prob"), "[", 2),
                 randomForest.formula = predict(model, newdata = .data, 
                                                type = "prob")[, positive],
                 ranger = predict(model, data = .data)$predictions[, positive],
                 xgb.Booster = predict(model, newdata = .data %>% 
                                         select(model$feature_names) %>% 
                                         data.matrix))

  names(pred) <- NULL

  pred_class <- ifelse(pred >= cutoff, positive, negative)
  
  if (is_factor) pred_class <- as.factor(pred_class)

  attr(pred_class , "target") <- target
  attr(pred_class , "positive") <- positive
  attr(pred_class , "negative") <- negative  
  attr(pred_class , "cutoff") <- cutoff
  attr(pred_class , "pred_prob") <- pred

  if (target %in% names(.data)) {
    attr(pred_class , "actual") <- .data[, target] %>% pull()
  }
    
  class(pred_class) <- append("predict_class", class(pred_class))

  pred_class
}

#' Predict binary classification model
#'
#' @description Predict some representative binary classification models.
#' @param .data A tbl_df. The data set to predict the model. It also supports tbl, and data.frame objects.
#' @param model A model_df. results of fitted model that created by run_models().
#' @param cutoff numeric. Cut-off that determines the positive from the probability of predicting the positive.
#'
#' @details Supported models are functions supported by the representative model package used in R environment.
#' The following binary classifications are supported:
#' \itemize{
#' \item "logistic" : logistic regression by predict.glm() in stats package.
#' \item "rpart" : recursive partitioning tree model by predict.rpart() in rpart package.
#' \item "ctree" : conditional inference tree model by predict() in stats package.
#' \item "randomForest" : random forest model by predict.randomForest() in randomForest package.
#' \item "ranger" : random forest model by predict.ranger() in ranger package.
#' }
#'
#' run_predict() is executed in parallel when predicting by model. 
#' However, it is not supported in MS-Windows operating system and RStudio environment.
#' 
#' @return model_df. results of predicted model.
#' model_df is composed of tbl_df and contains the following variables.:
#' \itemize{
#' \item step : character. The current stage in the model fit process. The result of calling run_predict() is returned as "2.Predicted".
#' \item model_id : character. Type of fit model.
#' \item target : character. Name of target variable.
#' \item is_factor : logical. Indicates whether the target variable is a factor.
#' \item positive : character. Level of positive class of binary classification.
#' \item negative : character. Level of negative class of binary classification.
#' \item fitted_model : list. Fitted model object.
#' \item predicted : list. Predicted value by individual model. Each value has a predict_class class object.
#' }
#'
#' @examples
#' library(dplyr)
#'
#' # Divide the train data set and the test data set.
#' sb <- rpart::kyphosis %>%
#'   split_by(Kyphosis)
#'
#' # Extract the train data set from original data set.
#' train <- sb %>%
#'   extract_set(set = "train")
#'
#' # Extract the test data set from original data set.
#' test <- sb %>%
#'   extract_set(set = "test")
#'
#' # Sampling for unbalanced data set using SMOTE(synthetic minority over-sampling technique).
#' train <- sb %>%
#'   sampling_target(seed = 1234L, method = "ubSMOTE")
#'
#' # Cleaning the set.
#' train <- train %>%
#'   cleanse
#'
#' # Run the model fitting.
#' result <- run_models(.data = train, target = "Kyphosis", positive = "present")
#' result
#'
#' # Predict the model.
#' pred <- run_predict(result, test)
#' pred
#'
#' # Run the several kinds model predict by dplyr
#' result %>%
#'   run_predict(test)
#'
#' @importFrom stats density
#' @export
run_predict <- function(model, .data, cutoff = 0.5) {
  if (dlookr::get_os() == "windows" || .Platform$GUI == "RStudio") {
    future::plan(future::sequential)
  } else {
    future::plan(future::multiprocess)
  }

  result <- purrr::map(seq(NROW(model)),
                       ~future::future(predictor(model$fitted_model[[.x]], .data,
                                                 model$target[[.x]],
                                                 model$positive[[.x]],
                                                 model$negative[[.x]],
                                                 model$is_factor[[.x]],
                                                 cutoff))) %>%
    tibble::tibble(step = "2.Predicted", model_id = model$model_id, target = model$target,
                   is_factor = model$is_factor, positive = model$positive, negative = model$negative, 
                   fitted_model = model$fitted_model, predicted = purrr::map(., ~future::value(.x)))

  result <- result[, -1]

  class(result) <- append("model_df", class(result))

  result
}
