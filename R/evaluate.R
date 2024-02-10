#' Compute Matthews Correlation Coefficient
#'
#' @description compute the Matthews correlation coefficient with actual and predict values.
#' @param predicted numeric. the predicted value of binary classification
#' @param y factor or character. the actual value of binary classification
#' @param positive level of positive class of binary classification
#'
#' @details The Matthews Correlation Coefficient has a value between -1 and 1, and the closer to 1,
#' the better the performance of the binary classification.
#'
#' @return numeric. The Matthews Correlation Coefficient.
#' @examples
#' # simulate actual data
#' set.seed(123L)
#' actual <- sample(c("Y", "N"), size = 100, prob = c(0.3, 0.7), replace = TRUE)
#' actual
#'
#' # simulate predict data
#' set.seed(123L)
#' pred <- sample(c("Y", "N"), size = 100, prob = c(0.2, 0.8), replace = TRUE)
#' pred
#'
#' # simulate confusion matrix
#' table(pred, actual)
#'
#' matthews(pred, actual, "Y")
#' @importFrom stats density
#' @export
matthews <- function (predicted, y, positive) {
  actual <- ifelse(y == positive, 1, 0)
  pred <- ifelse(predicted == positive, 1, 0)

  TP <- sum(actual == 1 & pred == 1)
  TN <- sum(actual == 0 & pred == 0)
  FP <- sum(actual == 0 & pred == 1)
  FN <- sum(actual == 1 & pred == 0)

  frac_up <- (TP * TN) - (FP * FN)
  frac_down <- as.double(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)

  if (any((TP + FP) == 0, (TP + FN) == 0, (TN + FP) == 0, (TN + FN) == 0))
    frac_down <- 1

  frac_up / sqrt(frac_down)
}

get_MCC <- function(predicted, y, positive, by = 0.01) {
  actual <- y %>% as.factor()

  cutoffs <- seq(from = 0, to = 1, by = by)

  get_matthews <- function(predicted, y, positive, thres) {
    pred <- ifelse(predicted >= thres, positive,
                   setdiff(unique(y), positive))

    matthews(pred, y, positive)
  }

  mcc <- sapply(cutoffs, function(x) get_matthews(predicted, y, positive, x))

  data.frame(prob = cutoffs, mcc = mcc)
}

get_cross <- function(predicted, y, positive) {
  pos <- predicted[y == positive]
  neg <- predicted[y != positive]

  density_pos <- stats::density(pos, from = 0, to = 1)
  density_neg <- stats::density(neg, from = 0, to = 1)

  diff_pos <- diff(density_pos$y) >= 0
  idx_pos <- which(diff(diff_pos) != 0) + 1

  diff_neg <- diff(density_neg$y) >= 0
  idx_neg <- which(diff(diff_neg) != 0) + 1

  idx <- sort(union(idx_pos, idx_neg))

  idx <- unlist(sapply(seq(idx)[-length(idx)],
    function(x) {
      pos <- density_pos$y[idx[x]:idx[x+1]]
      neg <- density_neg$y[idx[x]:idx[x+1]]

      idx[x] + which(diff(pos >= neg) != 0)
    }))

  list(x = round((density_pos$x[idx] + density_neg$x[idx]) / 2, 4),
       y = round((density_pos$y[idx] + density_neg$y[idx]) / 2, 4))
}


#' Visualization for cut-off selection
#'
#' @description plot_cutoff() visualizes a plot to select a cut-off that separates positive and
#' negative from the probabilities that are predictions of a binary classification,
#' and suggests a cut-off.
#' @param predicted numeric. the predicted value of binary classification
#' @param y factor or character. the actual value of binary classification
#' @param positive level of positive class of binary classification
#' @param type character. Visualization type. "mcc" draw the Matthews Correlation Coefficient scatter plot,
#' "density" draw the density plot of negative and positive,
#' and "prob" draws line or points plots of the predicted probability.
#' @param measure character. The kind of measure that calculates the cutoff.
#' "mcc" is the Matthews Correlation Coefficient, "cross" is the point where the positive
#' and negative densities cross, and "half" is the median of the probability, 0.5
#' @return numeric. cut-off value
#'
#' @details If the type argument is "prob", visualize the points plot if the number of observations
#' is less than 100. If the observation is greater than 100, draw a line plot.
#' In this case, the speed of visualization can be slow.
#'
#' @examples
#' library(ggplot2)
#' library(rpart)
#' data(kyphosis)
#'
#' fit <- glm(Kyphosis ~., family = binomial, kyphosis)
#' pred <- predict(fit, type = "response")
#'
#' cutoff <- plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "mcc")
#' cutoff
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "mcc", measure = "cross")
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "mcc", measure = "half")
#'
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "density", measure = "mcc")
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "density", measure = "cross")
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "density", measure = "half")
#'
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "prob", measure = "mcc")
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "prob", measure = "cross")
#' plot_cutoff(pred, kyphosis$Kyphosis, "present", type = "prob", measure = "half")
#' 
#' @import dplyr
#' @import ggplot2
#' @export
plot_cutoff <- function(predicted, y, positive, type = c("mcc", "density", "prob"),
  measure = c("mcc", "cross", "half")) {
  type <- match.arg(type)

  if (type == "mcc" & length(measure) == 3) {
    measure <- "mcc"
  } else if (type == "density" & length(measure) == 3) {
    measure <- "cross"
  } else if (type == "prob" & length(measure) == 3) {
    measure <- "prob"
  }

  if (type == "mcc" | measure == "mcc") {
    MCC <- get_MCC(predicted, y, positive)

    maxMCC <- MCC$mcc %>%
      max

    mcc <- MCC %>%
      filter(mcc == maxMCC) %>%
      dplyr::select(prob) %>%
      pull %>%
      max
  }

  if (type == "density" | measure == "cross") {
    cross <- get_cross(predicted, y, positive)

    cross_x <- cross$x
    cross_y <- cross$y
  }

  if (measure == "mcc") {
    cutoff <- mcc
  } else if (measure == "cross") {
    cutoff <- cross_x[length(cross_x)]
  } else if (measure == "half") {
    cutoff <- 0.5
  }

  if (type == "mcc") {
    p <- MCC %>%
      ggplot(aes(x = prob, y = mcc)) +
      geom_line(color = "blue") +
      annotate("pointrange", x = cutoff, y = maxMCC, ymin = 0, ymax = maxMCC,
        colour = "red", size = 0.5) +
      annotate("text", x = cutoff + 0.08, y = maxMCC, label = paste("cutoff", cutoff, sep = " = ")) +
      ylab("Matthews Correlation Coefficient (MCC)") +
      xlab("predictive probability") +
      ggtitle(label = "Probability vs MCC for choose cut-off",
        subtitle = paste("using measure", measure, sep = " : "))
  }

  if (type == "density") {
    dframe <- data.frame(actually = y, predicted = predicted)

    p <- ggplot(dframe, aes(x = predicted, colour = actually)) +
      geom_density() +
      geom_vline(xintercept = cutoff, colour = "blue", size = 0.5, linetype = "dashed") +
      annotate("text", x = cutoff + 0.08, y = 0.1, label = paste("cutoff", cutoff, sep = " = ")) +
      xlab("predictive probability") +
      ggtitle(label = "density for choose cut-off",
        subtitle = paste("using measure", measure, sep = " : "))
  }

  if (type == "prob") {
    idx <- order(predicted)
    unit <- length(idx) / 10

    if (length(y) <= 100) {
      target <- factor(ifelse(y[idx] == positive, "positive", "negative"),
        levels = c("positive", "negative"))

      p <- data.frame(prob = sort(predicted), target = target, idx = seq(target)) %>%
        ggplot(aes(x = idx, y = prob, color = target)) +
        geom_point() +
        geom_hline(yintercept = cutoff, linetype = 2)
    } else {
      idx_pos <- which(y[idx] == positive)
      updown <- predicted[idx[idx_pos]] >= cutoff

      target <- factor(ifelse(y[idx] == positive, "positive", "negative"),
        levels = c("positive", "negative"))

      p <- data.frame(prob = sort(predicted), target = target, idx = seq(target)) %>%
        ggplot(aes(x = idx, y = prob)) +
        geom_line(color = "gray45") +
        geom_hline(yintercept = cutoff)

      for (i in seq(idx_pos)) {
        p <- p + annotate("pointrange", x = idx_pos[i], y = cutoff,
          ymin = ifelse(updown[i], cutoff, 0), ymax = ifelse(updown[i], 1, cutoff),
          colour = c("blue", "red")[updown[i] + 1], size = 0.1)
      }

      p <- p + annotate("segment", x = 2 * unit, xend = 3 * unit, y = 1, yend = 1, colour = "red")
      p <- p + annotate("segment", x = 2 * unit, xend = 3 * unit, y = 0.95, yend = 0.95, colour = "blue")

      p <- p + annotate("text", x = c(unit, unit), y = c(1, 0.95),
        label = c("upper cutoff positive", "under cutoff positive"))
    }

    p <- p + annotate("text", x = unit / 2, y = cutoff + 0.05,
      label = paste("cutoff", cutoff, sep = " = ")) +
      ylab("fitted values") +
      xlab("index") +
      ggtitle(label = "probability for choose cut-off",
        subtitle = paste("using measure", measure, sep = " : "))
  }

  print(p)
  invisible(cutoff)
}


#' Calculate metrics for model evaluation
#'
#' @description Calculate some representative metrics for binary classification model evaluation.
#' @param pred numeric. Probability values that predicts the positive class of the target variable.
#' @param actual factor. The value of the actual target variable.
#' @param positive character. Level of positive class of binary classification.
#' @param metric character. The performance metrics you want to calculate. See details.
#' @param cutoff numeric. Threshold for classifying predicted probability values into positive and negative classes.
#' @param beta numeric. Weight of precision in harmonic mean for F-Beta Score.
#'
#' @details The cutoff argument applies only if the metric argument is "ZeroOneLoss", "Accuracy", "Precision", "Recall",
#' "Sensitivity", "Specificity", "F1_Score", "Fbeta_Score", "ConfusionMatrix".
#'
#' @return numeric or table object.
#' Confusion Matrix return by table object. and otherwise is numeric.:
#' The performance metrics calculated are as follows.:
#' \itemize{
#' \item ZeroOneLoss : Normalized Zero-One Loss(Classification Error Loss).
#' \item Accuracy : Accuracy.
#' \item Precision : Precision.
#' \item Recall : Recall.
#' \item Sensitivity : Sensitivity.
#' \item Specificity : Specificity.
#' \item F1_Score : F1 Score.
#' \item Fbeta_Score : F-Beta Score.
#' \item LogLoss : Log loss / Cross-Entropy Loss.
#' \item AUC : Area Under the Receiver Operating Characteristic Curve (ROC AUC).
#' \item Gini : Gini Coefficient.
#' \item PRAUC : Area Under the Precision-Recall Curve (PR AUC).
#' \item LiftAUC : Area Under the Lift Chart.
#' \item GainAUC : Area Under the Gain Chart.
#' \item KS_Stat : Kolmogorov-Smirnov Statistic.
#' \item ConfusionMatrix : Confusion Matrix.
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
#' # Calculate Accuracy.
#' performance_metric(attr(pred$predicted[[1]], "pred_prob"), test$Kyphosis,
#'   "present", "Accuracy")
#' # Calculate Confusion Matrix.
#' performance_metric(attr(pred$predicted[[1]], "pred_prob"), test$Kyphosis,
#'   "present", "ConfusionMatrix")
#' # Calculate Confusion Matrix by cutoff = 0.55.
#' performance_metric(attr(pred$predicted[[1]], "pred_prob"), test$Kyphosis,
#'   "present", "ConfusionMatrix", cutoff = 0.55)
#'    
#' @importFrom stats density
#' @export

performance_metric <- function(pred, actual, positive,
                               metric = c("ZeroOneLoss", "Accuracy", "Precision", "Recall", "Sensitivity",
                                          "Specificity", "F1_Score", "Fbeta_Score", "LogLoss", "AUC",
                                          "Gini", "PRAUC", "LiftAUC", "GainAUC", "KS_Stat",
                                          "ConfusionMatrix"),
                               cutoff = 0.5, beta = 1) {
  metric <- match.arg(metric)

  metric_factor <- c("ZeroOneLoss", "Accuracy", "Precision", "Recall",
                     "Sensitivity", "Specificity", "F1_Score", "Fbeta_Score",
                     "ConfusionMatrix")

  metric_0_1 <- c("LogLoss", "AUC", "Gini", "PRAUC", "LiftAUC", "GainAUC", "KS_Stat")

  if (metric %in% metric_factor) {
    level <- levels(factor(actual))
    pred_factor <- ifelse(pred < cutoff, setdiff(level, positive), positive)

    ZeroOneLoss <- mean(pred_factor != actual)
    Accuracy <- mean(pred_factor == actual)

    ConfusionMatrix <- table(predict = pred_factor, actual)

    idx_pos <- which(row.names(ConfusionMatrix) == positive)
    idx_neg <- which(row.names(ConfusionMatrix) != positive)

    TP <- ConfusionMatrix[idx_pos, idx_pos]
    TN <- ConfusionMatrix[idx_neg, idx_neg]
    FP <- ConfusionMatrix[idx_pos, idx_neg]
    FN <- ConfusionMatrix[idx_neg, idx_pos]

    Precision <- TP / (TP + FP)
    Recall <- Sensitivity <- TP / (TP + FN)
    Specificity <- TN / (TN + FP)
    F1_Score <- 2 * (Precision * Recall) / (Precision + Recall)
    Fbeta_Score <- (1 + beta ^ 2) * (Precision * Recall) /
      (beta ^ 2 * Precision + Recall)
  } else if (metric %in% metric_0_1) {
    actual_integer <- ifelse(actual == positive, 1, 0)

    LogLoss <- MLmetrics::LogLoss(pred, actual_integer)
    AUC <- MLmetrics::AUC(pred, actual_integer)
    Gini <- MLmetrics::Gini(pred, actual_integer)
    PRAUC <- MLmetrics::PRAUC(pred, actual_integer)
    LiftAUC <- MLmetrics::LiftAUC(pred, actual_integer)
    GainAUC <- MLmetrics::GainAUC(pred, actual_integer)
    KS_Stat <- MLmetrics::KS_Stat(pred, actual_integer)

    n_pos <- sum(actual_integer == 1)
    n_neg <- sum(actual_integer == 0)
  }

  get(metric)
}

#' Apply calculate performance metrics for model evaluation
#'
#' @description Apply calculate performance metrics for binary classification model evaluation.
#' @param model A model_df. results of predicted model that created by run_predict().
#' @param actual factor. A data of target variable to evaluate the model. It supports factor that has binary class.
#'
#' @return model_df. results of predicted model.
#' model_df is composed of tbl_df and contains the following variables.:
#' \itemize{
#' \item step : character. The current stage in the model fit process. The result of calling run_performance() is returned as "3.Performanced".
#' \item model_id : character. Type of fit model.
#' \item target : character. Name of target variable.
#' \item positive : character. Level of positive class of binary classification.
#' \item fitted_model : list. Fitted model object.
#' \item predicted : list. Predicted value by individual model. Each value has a predict_class class object.
#' \item performance : list. Calculate metrics by individual model. Each value has a numeric vector.
#' }
#' The performance metrics calculated are as follows.:
#' \itemize{
#' \item ZeroOneLoss : Normalized Zero-One Loss(Classification Error Loss).
#' \item Accuracy : Accuracy.
#' \item Precision : Precision.
#' \item Recall : Recall.
#' \item Sensitivity : Sensitivity.
#' \item Specificity : Specificity.
#' \item F1_Score : F1 Score.
#' \item Fbeta_Score : F-Beta Score.
#' \item LogLoss : Log loss / Cross-Entropy Loss.
#' \item AUC : Area Under the Receiver Operating Characteristic Curve (ROC AUC).
#' \item Gini : Gini Coefficient.
#' \item PRAUC : Area Under the Precision-Recall Curve (PR AUC).
#' \item LiftAUC : Area Under the Lift Chart.
#' \item GainAUC : Area Under the Gain Chart.
#' \item KS_Stat : Kolmogorov-Smirnov Statistic.
#' }
#'
#' @details 
#' run_performance() is performed in parallel when calculating the performance evaluation index. 
#' However, it is not supported in MS-Windows operating system and RStudio environment.
#' @examples
#' \donttest{
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
#' # Predict the model. (Case 1)
#' pred <- run_predict(result, test)
#' pred
#'
#' # Calculate performace metrics. (Case 1)
#' perf <- run_performance(pred)
#' perf
#' perf$performance
#'
#' # Predict the model. (Case 2)
#' pred <- run_predict(result, test[, -1])
#' pred
#'
#' # Calculate performace metrics. (Case 2)
#' perf <- run_performance(pred, pull(test[, 1]))
#' perf
#' perf$performance
#' 
#' # Convert to matrix for compare performace.
#' sapply(perf$performance, "c")
#' }
#' 
#' @importFrom stats density
#' @importFrom parallelly supportsMulticore
#' @importFrom future plan
#' @export
run_performance <- function(model, actual = NULL) {
  metric <- list("ZeroOneLoss", "Accuracy", "Precision", "Recall",
                 "Sensitivity", "Specificity", "F1_Score", "Fbeta_Score", "LogLoss",
                 "AUC", "Gini", "PRAUC", "LiftAUC", "GainAUC", "KS_Stat")

  performance <- function(pred, actual, positive) {
    pmetric <- sapply(metric, function(x) performance_metric(pred, actual, positive, x))
    names(pmetric) <- metric

    pmetric
  }

  if (dlookr::get_os() == "windows" || .Platform$GUI == "RStudio") {
    future::plan(future::sequential)
  } else {
    if (parallelly::supportsMulticore()) {
      oplan <- future::plan(future::multicore)
    } else {
      oplan <- future::plan(future::multisession)
    }
    on.exit(future::plan(oplan))
  }
  
  if (is.null(actual)) {
    result <- purrr::map(seq(NROW(model)),
                         ~future::future(performance(attr(pred$predicted[[.x]], "pred_prob"),
                                                     attr(pred$predicted[[.x]], "actual"),
                                                     attr(pred$predicted[[.x]], "positive")),
                                         seed = TRUE)) %>%
      tibble::tibble(step = "3.Performanced", model_id = model$model_id, target = model$target,
                     positive = model$positive, fitted_model = model$fitted_model,
                     predicted = model$predicted,
                     performance = purrr::map(., ~future::value(.x)))
  } else {
    result <- purrr::map(seq(NROW(model)),
                         ~future::future(performance(attr(pred$predicted[[.x]], "pred_prob"),
                                                     actual,
                                                     attr(pred$predicted[[.x]], "positive")),
                                         seed = TRUE)) %>%
      tibble::tibble(step = "3.Performanced", model_id = model$model_id, target = model$target,
                     positive = model$positive, fitted_model = model$fitted_model,
                     predicted = model$predicted,
                     performance = purrr::map(., ~future::value(.x)))
  }

  result <- result[, -1]

  class(result) <- append("model_df", class(result))

  result
}


#' Compare model performance
#'
#' @description compare_performance() compares the performance of a model with several model performance metrics.
#' @param model A model_df. results of predicted model that created by run_predict().
#' @return list. results of compared model performance.
#' list has the following components:
#' \itemize{
#' \item recommend_model : character. The name of the model that is recommended as the best among the various models.
#' \item top_count : numeric. The number of best performing performance metrics by model.
#' \item mean_rank : numeric. Average of ranking individual performance metrics by model.
#' \item top_metric : list. The name of the performance metric with the best performance on individual performance metrics by model.
#' }
#' The performance metrics calculated are as follows.:
#' \itemize{
#' \item ZeroOneLoss : Normalized Zero-One Loss(Classification Error Loss).
#' \item Accuracy : Accuracy.
#' \item Precision : Precision.
#' \item Recall : Recall.
#' \item Specificity : Specificity.
#' \item F1_Score : F1 Score.
#' \item LogLoss : Log loss / Cross-Entropy Loss.
#' \item AUC : Area Under the Receiver Operating Characteristic Curve (ROC AUC).
#' \item Gini : Gini Coefficient.
#' \item PRAUC : Area Under the Precision-Recall Curve (PR AUC).
#' \item LiftAUC : Area Under the Lift Chart.
#' \item GainAUC : Area Under the Gain Chart.
#' \item KS_Stat : Kolmogorov-Smirnov Statistic.
#' }
#'
#' @examples
#' \donttest{
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
#'
#' # Predict the model.
#' pred <- run_predict(result, test)
#'
#' # Compare the model performance
#' compare_performance(pred)
#'}
#'
#' @export
#'
compare_performance <- function(model) {
  perf <- run_performance(model)
  performance <- sapply(perf$performance, "c")

  metric <- list("ZeroOneLoss", "Accuracy", "Precision", "Recall",
                 "Specificity", "F1_Score", "LogLoss", "AUC", "Gini",
                 "PRAUC", "LiftAUC", "GainAUC", "KS_Stat")

  performance <- performance[rownames(performance) %in% metric, ]
  colnames(performance) <- pred$model_id

  func <- c("min", "max", "max", "max", "max", "max",
            "min", "max", "max", "max", "max", "max", "max")

  get_cond <- function(pos, func) {
    do.call(func, list(performance[pos, ]))
  }

  get_rank <- function(pos, func) {
    if (func == "min")
      rank(performance[pos, ])
    else
      rank(100 - performance[pos, ])
  }

  top <- mapply(get_cond, seq(nrow(performance)), func)
  perf_logical <- sweep(performance, 1, top, "==")
  counts <- apply(perf_logical, 2, sum)

  pos <- lapply(as.data.frame(perf_logical), which)
  top_metric <- lapply(pos, function(x) unlist(metric[x]))

  ranks <- mapply(get_rank, seq(nrow(performance)), func)
  ranks <- apply(ranks, 1, mean)

  recommend <- unique(names(which.max(counts)), names(which.min(ranks)))

  list(recommend_model = recommend, top_metric_count = counts,
       mean_rank = ranks, top_metric = top_metric)
}


#' Visualization for ROC curve
#'
#' @description plot_performance() visualizes a plot to ROC curve that separates model algorithm.
#' @param model A model_df. results of predicted model that created by run_predict().
#'
#' @details The ROC curve is output for each model included in the model_df class object specified as a model argument.
#' @return There is no return value. Only the plot is drawn.
#' @examples
#' \donttest{
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
#'
#' # Predict the model.
#' pred <- run_predict(result, test)
#'
#' # Plot ROC curve
#' plot_performance(pred)
#' }
#' 
#' @import ggplot2
#' @export
plot_performance <- function(model) {
  model_id <- model$model_id

  get_prediction <- function(x) {
    pred_prob <- attr(x, "pred_prob")
    actual <- attr(x, "actual")
    positive <- attr(x, "positive")

    pred <- ROCR::prediction(pred_prob, ifelse(actual == positive, 1, 0))
    perf <- ROCR::performance(pred, "tpr", "fpr" )
    data.frame(x = unlist(perf@x.values), y = unlist(perf@y.values))
  }

  tmp <- lapply(model$predicted, get_prediction)

  data_all <- NULL
  for (i in seq(length(tmp))) {
    data_all <- rbind(data_all, data.frame(tmp[[i]], model_id = model_id[i]))
  }

  color_palette <- c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
                     "#E6AB02", "#A6761D")
  color_palette <- setNames(color_palette[seq(model_id)], model_id)

  ggplot(data_all, aes(x = x, y = y, color = model_id)) +
    geom_line() +
    geom_line(size = 1.5, alpha = 0.6) +
    ggtitle("ROC curve") +
    xlab("False Positive Ratio (1-Specificity)") +
    ylab("True Positive Ratio (Sensitivity)") +
    scale_colour_manual(values = color_palette) +
    geom_abline(slope = 1, intercept = 0, lty = 2, colour = 'black')
}
