#' Extract the data to fit the model
#'
#' @description To solve the imbalanced class, perform sampling in the train set of split_df.
#' @details In order to solve the problem of imbalanced class, sampling is performed by under sampling,
#' over sampling, SMOTE method.
#' @section attributes of train_df class:
#' The attributes of the train_df class are as follows.:
#'
#' \itemize{
#' \item sample_seed : integer. random seed used for sampling
#' \item method : character. sampling methods.
#' \item perc : integer. perc argument value
#' \item k : integer. k argument value
#' \item perc.over : integer. perc.over argument value
#' \item perc.under : integer. perc.under argument value
#' \item binary : logical. whether the target variable is a binary class
#' \item target : character. target variable name
#' \item minority : character. the level of the minority class
#' \item majority : character. the level of the majority class
#' }
#'
#' @param .data an object of class "split_df", usually, a result of a call to split_df().
#' @param method character. sampling methods. "ubUnder" is under-sampling,
#' and "ubOver" is over-sampling, "ubSMOTE" is SMOTE(Synthetic Minority Over-sampling TEchnique).
#' @param seed integer. random seed used for sampling
#' @param perc integer. The percentage of positive class in the final dataset.
#' It is used only in under-sampling. The default is 50. perc can not exceed 50.
#' @param k integer. It is used only in over-sampling and SMOTE.
#' If over-sampling and if K=0: sample with replacement from the minority class until
#' we have the same number of instances in each class. under-sampling and if K>0:
#' sample with replacement from the minority class until we have k-times
#' the original number of minority instances.
#' If SMOTE, the number of neighbours to consider as the pool from where the new
#' examples are generated
#' @param perc.over integer. It is used only in SMOTE. per.over/100 is the number of new instances
#' generated for each rare instance. If perc.over < 100 a single instance is generated.
#' @param perc.under integer. It is used only in SMOTE. perc.under/100 is the number
#' of "normal" (majority class) instances that are randomly selected for each smoted
#' observation.
#' @return An object of train_df.
#' @export
#' @examples
#' library(dplyr)
#'
#' # Credit Card Default Data
#' head(ISLR::Default)
#'
#' # Generate data for the example
#' sb <- ISLR::Default %>%
#'   split_by(default)
#'
#' # under-sampling with random seed
#' under <- sb %>%
#'   sampling_target(seed = 1234L)
#'
#' under %>%
#'   count(default)
#'
#' # under-sampling with random seed, and minority class frequency is 40%
#' under40 <- sb %>%
#'   sampling_target(seed = 1234L, perc = 40)
#'
#' under40 %>%
#'   count(default)
#'
#' # over-sampling with random seed
#' over <- sb %>%
#'   sampling_target(method = "ubOver", seed = 1234L)
#'
#' over %>%
#'   count(default)
#'
#' # over-sampling with random seed, and k = 10
#' over10 <- sb %>%
#'   sampling_target(method = "ubOver", seed = 1234L, k = 10)
#'
#' over10 %>%
#'   count(default)
#'
#' # SMOTE with random seed
#' smote <- sb %>%
#'   sampling_target(method = "ubSMOTE", seed = 1234L)
#'
#' smote %>%
#'   count(default)
#'
#' # SMOTE with random seed, and perc.under = 250
#' smote250 <- sb %>%
#'   sampling_target(method = "ubSMOTE", seed = 1234L, perc.under = 250)
#'
#' smote250 %>%
#'   count(default)
#'
#' @importFrom tidyselect vars_select
#' @importFrom rlang quos
#' @importFrom unbalanced ubUnder ubOver ubSMOTE ubOSS
#' @export
#'
sampling_target <- function(.data, method = c("ubUnder", "ubOver", "ubSMOTE"),
  seed = NULL, perc = 50, k = ifelse(method == "ubSMOTE", 5, 0),
  perc.over = 200, perc.under = 200) {
  if(is(.data) != "split_df") {
    stop("x is not split_df class")
  }

  target <- attr(.data, "target")
  target_idx <- .data %>%
    ungroup() %>%
    dplyr::select(-split_flag) %>%
    names(.) %in% target %>%
    which

  method <- match.arg(method)

  minority <- attr(.data, "minority")
  majority <- attr(.data, "majority")

  Y <- .data %>%
    filter(split_flag == "train") %>%
    ungroup() %>%
    dplyr::select(target = target_idx) %>%
    mutate(target = ifelse(target == minority, "1", "0") %>%
        factor(levels = c("1", "0"))) %>%
    pull()

  X <- .data %>%
    filter(split_flag == "train") %>%
    ungroup() %>%
    dplyr::select(-target, -split_flag)

  type_before <- sapply(X, function(x) is(x)[1])

  if (method %in% c("ubSMOTE")) {
    idx_ordered <- which(type_before == "ordered")

    if (length(idx_ordered) > 0) {
      for (i in idx_ordered) {
        X[, i] <- factor(pull(X[, i]), ordered = FALSE)
      }
    }

    idx_character <- which(type_before == "character")

    if (length(idx_character) > 0) {
      for (i in idx_character) {
        X[, i] <- factor(pull(X[, i]))
      }
    }
  }

  if (is.null(seed))
    seed <- sample(seq(1e5), size = 1)

  set.seed(seed)

  samples <- switch (method,
    ubUnder = unbalanced::ubUnder(X, Y, perc = perc, method = "percPos", w = NULL),
    ubOver  = unbalanced::ubOver(X, Y, k = k),
    ubOSS  = unbalanced::ubOSS(X, Y),
    ubSMOTE = unbalanced::ubSMOTE(data.frame(X), Y, perc.over = perc.over, k = k,
      perc.under = perc.under)
  )

  if (method == "ubUnder") {
    result <- .data %>%
      filter(split_flag == "train") %>%
      ungroup() %>%
      dplyr::select(-split_flag) %>%
      .[-c(samples$id.rm), ]

  } else if(method %in% c("ubOver", "ubSMOTE")) {
    Y <- factor(ifelse(samples$Y == 1, minority, majority))

    if (method == "ubSMOTE") {
      type_after <- sapply(samples$X, function(x) is(x)[1])

      change_int <- which((type_after != type_before) & (type_before == "integer"))

      if (length(change_int) > 0) {
        for (i in change_int) {
          samples$X[, i] <- as.integer(samples$X[, i])
        }
      }

      if (length(idx_ordered) > 0) {
        for (i in idx_ordered) {
          samples$X[, i] <- factor(samples$X[, i], ordered = TRUE)
        }
      }

      if (length(idx_character) > 0) {
        for (i in idx_character) {
          samples$X[, i] <- as.character(samples$X[, i])
        }
      }
    }

    #result <- samples$X %>%
    #  mutate_at(attr(.data, "target"), function(x) Y) %>%
    #  as.tbl()

    smpl <- cbind(samples$X, Y)
    names(smpl)[NCOL(smpl)] <- attr(.data, "target")

    result <- smpl %>%
      mutate_at(attr(.data, "target"), function(x) Y) %>%
      as.tbl()

    if (target_idx == 1) {
      result <- result %>%
        dplyr::select(NCOL(result), 1:NCOL(samples$X))
    } else if (target_idx == NCOL(result)) {
      result <- result %>%
        dplyr::select(1:NCOL(samples$X), NCOL(result))
    } else {
      result <- result %>%
        dplyr::select(1:(target_idx - 1), NCOL(result),
          (target_idx + 1):NCOL(samples$X))
    }

    names(result)[target_idx] <- target
  }

  attr(result , "sample_seed") <- seed
  attr(result , "method") <- method
  attr(result , "perc") <- perc
  attr(result , "k") <- k
  attr(result , "perc.over") <- perc.over
  attr(result , "perc.under") <- perc.under
  attr(result , "binary") <- attr(.data , "binary")
  attr(result , "target") <- attr(.data , "target")
  attr(result , "minority") <- attr(.data , "minority")
  attr(result , "majority") <- attr(.data , "majority")

  class(result) <- append("train_df", class(result))

  result
}
