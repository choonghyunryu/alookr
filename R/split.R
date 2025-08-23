#' @rdname split_by.data.frame
#' @export
split_by <- function(.data, ...) {
  UseMethod("split_by", .data)
}


#' Split Data into Train and Test Set
#'
#' @description The split_by() splits the data.frame or tbl_df into a train set and a test set.
#'
#' @details The split_df class is created, which contains the split information and criteria to separate the training and the test set.
#'
#' @section attributes of split_by:
#' The attributes of the split_df class are as follows.:
#'
#' \itemize{
#' \item split_seed : integer. random seed used for splitting
#' \item target : character. the name of the target variable
#' \item binary : logical. whether the target variable is binary class
#' \item minority : character. the name of the minority class
#' \item majority : character. the name of the majority class
#' \item minority_rate : numeric. the rate of the minority class
#' \item majority_rate : numeric. the rate of the majority class
#' }
#'
#' @param .data a data.frame or a \code{\link[tibble]{tbl_df}}.
#' @param target unquoted expression or variable name. the name of the target variable
#' @param ratio numeric. the ratio of the train dataset. default is 0.7
#' @param seed random seed used for splitting
#' @param ... further arguments passed to or from other methods.
#'
#' @return An object of split_by.
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
#' sb
#'
#' @method split_by data.frame
#' @importFrom tidyselect vars_select
#' @importFrom rlang quos
#' @export
split_by.data.frame <- function(.data, target, ratio = 0.7, seed = NULL, ...) {
  tryCatch(target <- tidyselect::vars_select(names(.data), !! rlang::enquo(target)),
    error = function(e) {
      pram <- as.character(substitute(target))
      stop(sprintf("Column %s is unknown", pram))
    }, finally = NULL)

  split_by_impl(.data, target, ratio, seed)
}


#' @import dplyr
#' @importFrom caTools sample.split
split_by_impl <- function(.data, target, ratio, seed = NULL) {
  if (!target %in% names(.data)) {
    stop(sprintf("%s in not variable in %s", target,
      as.character(substitute(.data))))
  }

  if (is.null(seed))
    seed <- sample(seq(1e5), size = 1)

  set.seed(seed)
  flag <- .data %>%
    dplyr::select(1) %>%
    pull %>%
    caTools::sample.split(SplitRatio = ratio)

  .data <- .data %>%
    mutate(split_flag = ifelse(flag, "train", "test"))

  split_df <- grouped_df(.data, "split_flag")

  tabs <- table(.data[, target])

  attr(split_df , "split_seed") <- seed
  attr(split_df , "target") <- target
  attr(split_df , "binary") <- length(tabs) == 2
  attr(split_df , "minority") <- names(tabs[tabs == min(tabs)])
  attr(split_df , "majority") <- names(tabs[tabs == max(tabs)])
  attr(split_df , "minority_rate") <- tabs[names(tabs[tabs == min(tabs)])] /sum(tabs)
  attr(split_df , "majority_rate") <- tabs[names(tabs[tabs == max(tabs)])] /sum(tabs)

  class(split_df) <- append("split_df", class(split_df))

  split_df
}


#' Summarizing split_df information
#'
#' @description summary method for "split_df" class.
#' @param object an object of class "split_df", usually, a result of a call to split_df().
#' @param ... further arguments passed to or from other methods.
#' @details
#' summary.split_df provides information on the number of two split data sets, minority class and majority class.
#'
#' @return NULL is returned. 
#' However, the split train set and test set information are displayed. The output information is as follows.:
#' 
#' \itemize{
#' \item Random seed
#' \item Number of train sets and test sets
#' \item Name of target variable
#' \item Target variable minority class and majority class information (label and ratio)
#' }
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
#' sb
#' summary(sb)
#'
#' @method summary split_df
#' @export
summary.split_df <- function(object, ...) {
  split_seed <- attr(object, "split_seed")
  target <- attr(object, "target")
  binary <- attr(object, "binary")
  minority <- attr(object, "minority")
  majority <- attr(object, "majority")
  minority_rate <- attr(object, "minority_rate")
  majority_rate <- attr(object, "majority_rate")

  counts <- object %>%
    count(split_flag) %>%
    ungroup() %>%
    dplyr::select(n) %>%
    pull

  cat("** Split train/test set information **\n")
  cat(" + random seed        : ", split_seed, "\n")
  cat(" + split data           ", "\n")
  cat("    - train set count : ", counts[2], "\n")
  cat("    - test set count  : ", counts[1], "\n")
  cat(" + target variable    : ", target, "\n")
  cat(sprintf("    - minority class  :  %s (%f)\n", minority, minority_rate))
  cat(sprintf("    - majority class  :  %s (%f)\n", majority, majority_rate))
}


#' Comparison of numerical variables of train set and test set
#'
#' @description Compare the statistics of the numerical variables of
#' the train set and test set included in the "split_df" class.
#' @param .data an object of class "split_df", usually, a result of a call to split_df().
#' @param ... one or more unquoted expressions separated by commas.
#' Select the numeric variable you want to compare.
#' You can treat variable names like they are positions.
#' Positive values select variables; negative values to drop variables.
#' If the first expression is negative, compare_target_numeric() will automatically
#' start with all variables.
#' These arguments are automatically quoted and evaluated in a context where column names
#' represent column positions.
#' They support unquoting and splicing.
#'
#' @details Compare the statistics of the numerical variables of the train set and
#' the test set to determine whether the raw data is well separated into two data sets.
#' @return tbl_df.
#' Variables for comparison:
#' \itemize{
#' \item variable : character. numeric variable name
#' \item train_mean : numeric. arithmetic mean of train set
#' \item test_mean : numeric. arithmetic mean of test set
#' \item train_sd : numeric. standard deviation of train set
#' \item test_sd : numeric. standard deviation of test set
#' \item train_z : numeric. the arithmetic mean of the train set divided by
#' the standard deviation
#' \item test_z : numeric. the arithmetic mean of the test set divided by
#' the standard deviation
#' }
#'
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
#' sb %>%
#'   compare_target_numeric()
#'
#' sb %>%
#'   compare_target_numeric(balance)
#'
#' @importFrom tidyselect vars_select
#' @importFrom rlang quos
#' @importFrom tidyr gather extract unite spread
#' @importFrom methods is
#' @importFrom stats sd
#' @import dplyr
#' @export
compare_target_numeric <- function(.data, ...) {
  vars <- tidyselect::vars_select(names(.data), ...)
  if (length(vars) == 0) vars <- names(.data)

  if(is(.data) != "split_df") {
    stop(".data is not split_df class")
  }

  z <- function(x, na.rm = TRUE) {
    mean(x, na.rm = na.rm) / sd(x, na.rm = na.rm)
  }

  df <- .data %>%
    select(vars, split_flag) %>%
    select_if(is.numeric)

  if (NCOL(df) == 1) stop("numeric variables not specified")

  if (NCOL(df) > 2) {
    df %>%
      group_by(split_flag) %>%
      summarise_all(c("mean", "sd", "z"), na.rm = TRUE) %>%
      tidyr::gather(variable, value, -split_flag) %>%
      tidyr::extract(variable, into = c("variable", "metric"), "(.*)[_](.*)$") %>%
      tidyr::unite(temp, split_flag, metric) %>%
      tidyr::spread(temp, value) %>%
      dplyr::select(variable, train_mean, test_mean,
        train_sd, test_sd, train_z, test_z)
  } else{
    df <- df %>%
      group_by(split_flag) %>%
      summarise_all(c("mean", "sd", "z"), na.rm = TRUE)

    names(df)[2:4] <- paste(vars, names(df)[2:4], sep = "_")

    df %>%
      tidyr::gather(variable, value, -split_flag) %>%
      tidyr::extract(variable, into = c("variable", "metric"), "(.*)[_](.*)$") %>%
      tidyr::unite(temp, split_flag, metric) %>%
      tidyr::spread(temp, value) %>%
      dplyr::select(variable, train_mean, test_mean,
        train_sd, test_sd, train_z, test_z)
  }

}


#' Comparison of categorical variables of train set and test set
#'
#' @description Compare the statistics of the categorical variables of
#' the train set and test set included in the "split_df" class.
#' @param .data an object of class "split_df", usually, a result of a call to split_df().
#' @param add_character logical. Decide whether to include text variables in the
#' compare of categorical data. The default value is FALSE, which also not includes character variables.
#' @param margin logical. Choose to calculate the marginal frequency information.
#' @param ... one or more unquoted expressions separated by commas.
#' Select the categorical variable you want to compare.
#' You can treat variable names like they are positions.
#' Positive values select variables; negative values to drop variables.
#' If the first expression is negative, compare_target_category() will automatically
#' start with all variables.
#' These arguments are automatically quoted and evaluated in a context where column names
#' represent column positions.
#' They support unquoting and splicing.
#'
#' @details Compare the statistics of the numerical variables of the train set and
#' the test set to determine whether the raw data is well separated into two data sets.
#' @return tbl_df.
#' Variables of tbl_df for comparison:
#' \itemize{
#' \item variable : character. categorical variable name
#' \item level : factor. level of categorical variables
#' \item train : numeric. the relative frequency of the level in the train set
#' \item test : numeric. the relative frequency of the level in the test set
#' \item abs_diff : numeric. the absolute value of the difference between two
#' relative frequencies
#' }
#'
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
#' sb %>%
#'   compare_target_category()
#'
#' sb %>%
#'   compare_target_category(add_character = TRUE)
#'
#' sb %>%
#'   compare_target_category(margin = TRUE)
#'
#' sb %>%
#'   compare_target_category(student)
#'
#' sb %>%
#'   compare_target_category(student, margin = TRUE)
#'
#' @importFrom tidyselect vars_select
#' @importFrom rlang quos
#' @importFrom tidyr spread
#' @importFrom dlookr find_class
#' @importFrom methods is
#' @import dplyr
#' @export
compare_target_category <- function(.data, ..., add_character = FALSE, margin = FALSE) {
  if(is(.data) != "split_df") {
    stop(".data is not split_df class")
  }

  vars <- tidyselect::vars_select(names(.data), ...)
  if (length(vars) == 0) vars <- names(.data)

  if (length(vars) == 1 & !tibble::is_tibble(.data))
    .data <- tibble::as_tibble(.data)

  vars <- setdiff(vars, "split_flag")

  if (add_character)
    idx_factor <- dlookr::find_class(.data[, vars], type = "categorical2")
  else
    idx_factor <- dlookr::find_class(.data[, vars], type = "categorical")

  relative_table <- function(df, var, margin) {
    tab <- df %>%
      dplyr::select(split_flag, .level = var) %>%
      count(split_flag, .level) %>%
      group_by(split_flag) %>%
      mutate(relative = n / sum(n) * 100) %>%
      dplyr::select(-n) %>%
      tidyr::spread(split_flag, relative) %>%
      mutate(variable = var, abs_diff = abs(train - test)) %>%
      dplyr::select(variable, level = .level, train, test, abs_diff)

    if (margin) {
      tab <- rbind(tab, tribble(~variable, ~level, ~train, ~test, ~abs_diff,
        var, "<Total>", 100, 100, sum(tab$abs_diff)))
    }
    tab
  }

  suppressWarnings(
    result <- lapply(vars[idx_factor],
                     function(x) relative_table(.data, x, margin))
  )

  suppressWarnings(
    do.call("rbind", result)
  )
}


#' Comparison plot of train set and test set
#'
#' @description Plot compare information of the train set and test set included
#' in the "split_df" class.
#' @param .data an object of class "split_df", usually, a result of a call to split_df().
#' @param ... one or more unquoted expressions separated by commas.
#' Select the variable you want to plotting.
#' You can treat variable names like they are positions.
#' Positive values select variables; negative values to drop variables.
#' If the first expression is negative, compare_target_category() will automatically
#' start with all variables.
#' These arguments are automatically quoted and evaluated in a context where column names
#' represent column positions.
#' They support unquoting and splicing.
#'
#' @details The numerical variables are density plots and the categorical variables are
#' mosaic plots to compare the distribution of train sets and test sets.
#'
#' @return There is no return value. Draw only the plot.
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
#' sb %>%
#'   compare_plot("income")
#'
#' sb %>%
#'   compare_plot()
#' @importFrom tidyselect vars_select
#' @importFrom rlang quos
#' @import dplyr
#' @import ggplot2
#' @import ggmosaic
#' @export
compare_plot <- function(.data, ...) {
  vars <- tidyselect::vars_select(names(.data), ...)
  if (length(vars) == 0) vars <- names(.data)

  plot_numeric <- function(df, var) {
    df %>%
      ggplot(aes_string(x = var, colour = "split_flag")) +
      geom_density() +
      ggtitle(label = "Density of Train Set vs Test Set",
        subtitle = paste("variable", var, sep = ":")) +
      xlab(var)
  }

  plot_category <- function(df, var) {
    df %>%
      dplyr::select(split_flag, variable = var) %>%
      ungroup() %>%
      mutate(split_flag = ordered(split_flag, levels = c("train", "test"))) %>%
      ggplot() +
      ggmosaic::geom_mosaic(aes(x = ggmosaic::product(split_flag), fill = variable)) +
      ggmosaic::scale_x_productlist("dataset class", labels = c("Training set", "Test set")) +
      ggtitle(label = "Frequency of Train Set vs Test Set",
        subtitle = paste("variable", var, sep = ":"))
  }

  plot_compare <- function(df, var) {
    flag_factor <- sb[, var] %>%
      pull %>%
      is.factor

    flag_numeric <- sb[, var] %>%
      pull %>%
      is.numeric

    if (flag_numeric) print(plot_numeric(df, var))
    if (flag_factor) print(plot_category(df, var))
  }

  tmp <- lapply(vars,
    function(x) plot_compare(.data, x))
}



#' Diagnosis of train set and test set of split_df object
#'
#' @description Diagnosis of similarity between datasets splitted by train set and set included in the "split_df" class.
#' @param .data an object of class "split_df", usually, a result of a call to split_df().
#' @param add_character logical. Decide whether to include text variables in the
#' compare of categorical data. The default value is FALSE, which also not includes character variables.
#' @param uniq_thres numeric. Set a threshold to removing variables when the ratio of unique values(number of unique values / number of observation) is greater than the set value.
#' @param miss_msg logical. Set whether to output a message when diagnosing missing value.
#' @param verbose logical. Set whether to echo information to the console at runtime.
#'
#' @details In the two split datasets, a variable with a single value, a variable with a level not found in any dataset, and a variable with a high ratio to the number of levels are diagnosed.
#'
#' @return list.
#' Variables of tbl_df for first component named "single_value":
#' \itemize{
#' \item variables : character. variable name
#' \item train_uniq : character. the type of unique value in train set. it is divided into "single" and "multi".
#' \item test_uniq : character. the type of unique value in test set. it is divided into "single" and "multi".
#' }
#'
#' Variables of tbl_df for second component named "uniq_rate":
#' \itemize{
#' \item variables : character. categorical variable name
#' \item train_uniqcount : numeric. the number of unique value in train set
#' \item train_uniqrate : numeric. the ratio of unique values(number of unique values / number of observation) in train set
#' \item test_uniqcount : numeric. the number of unique value in test set
#' \item test_uniqrate : numeric. the ratio of unique values(number of unique values / number of observation) in test set
#' }
#'
#' Variables of tbl_df for third component named "missing_level":
#' \itemize{
#' \item variables : character. variable name
#' \item n_levels : integer. count of level of categorical variable
#' \item train_missing_nlevel : integer. the number of non-existent levels in the train set
#' \item test_missing_nlevel : integer. he number of non-existent levels in the test set
#' }
#'
#' @examples
#' library(dplyr)
#'
#' # Credit Card Default Data
#' head(ISLR::Default)
#'
#' defaults <- ISLR::Default
#' defaults$id <- seq(NROW(defaults))
#'
#' set.seed(1)
#' defaults[sample(seq(NROW(defaults)), 3), "student"] <- NA
#' set.seed(2)
#' defaults[sample(seq(NROW(defaults)), 10), "balance"] <- NA
#'
#' sb <- defaults %>%
#'   split_by(default)
#'
#' sb %>%
#'   compare_diag()
#'
#' sb %>%
#'   compare_diag(add_character = TRUE)
#'
#' sb %>%
#'   compare_diag(uniq_thres = 0.0005)
#'
#' @importFrom tidyr extract
#' @importFrom dlookr diagnose
#' @importFrom tibble tribble
#' @importFrom methods is
#' @import dplyr
#' @export
compare_diag <- function(.data, add_character = FALSE, uniq_thres = 0.01,
  miss_msg = TRUE, verbose = TRUE) {
  if(is(.data) != "split_df") {
    stop(".data is not split_df class")
  }

  vars <- names(.data)

  if (length(vars) == 1 & !tibble::is_tibble(.data))
    .data <- tibble::as_tibble(.data)

  suppressMessages(
    missing_value <- .data %>%
      extract_set(set = "train") %>%
      dlookr::diagnose() %>%
      filter(missing_count > 0) %>%
      mutate(train_misscount = missing_count) %>%
      mutate(train_missrate = missing_percent) %>%
      dplyr::select(variables, train_misscount, train_missrate) %>%
      full_join(.data %>%
          extract_set(set = "test") %>%
          dlookr::diagnose() %>%
            filter(missing_count > 0) %>%
            mutate(train_misscount = missing_count) %>%
            mutate(test_missrate = missing_percent) %>%
            dplyr::select(variables, train_misscount, test_missrate)
      )
  )

  if (nrow(missing_value) & miss_msg) {
    vars <- missing_value %>%
      select(variables) %>%
      pull
    message("* Detected diagnose missing value")
    message(paste(" -", vars, collapse = "\n"))
  }

  suppressMessages(
    single_value <- .data %>%
      extract_set(set = "train") %>%
      diagnose() %>%
      dplyr::select(variables, unique_count) %>%
      filter(unique_count == 1) %>%
      mutate(train_uniq = "sigle") %>%
      full_join(.data %>%
          extract_set(set = "test") %>%
          diagnose() %>%
          dplyr::select(variables, unique_count) %>%
          filter(unique_count == 1) %>%
          mutate(test_uniq = "sigle")) %>%
      dplyr::select(-unique_count) %>%
      mutate(train_uniq = ifelse(is.na(train_uniq), "multi", "single")) %>%
      mutate(test_uniq = ifelse(is.na(test_uniq), "multi", "single"))
  )

  if (nrow(single_value)) {
    vars <- single_value %>%
      select(variables) %>%
      pull
    message("* Detected diagnose single unique value")
    message(paste(" -", vars, collapse = "\n"))
  }

  suppressMessages(
    uniq_rate <- .data %>%
      extract_set(set = "train") %>%
      dlookr::diagnose() %>%
      filter(types %in% c("character", "factor", "ordered")) %>%
      filter(unique_rate >= uniq_thres) %>%
      mutate(train_uniqcount = unique_count) %>%
      mutate(train_uniqrate = unique_rate) %>%
      dplyr::select(variables, train_uniqcount, train_uniqrate) %>%
      full_join(.data %>%
          extract_set(set = "test") %>%
          dlookr::diagnose() %>%
            filter(types %in% c("character", "factor", "ordered")) %>%
            filter(unique_rate >= uniq_thres) %>%
            mutate(test_uniqcount = unique_count) %>%
            mutate(test_uniqrate = unique_rate) %>%
            dplyr::select(variables, test_uniqcount, test_uniqrate)
      )
  )

  if (nrow(uniq_rate)) {
    vars <- uniq_rate %>%
      select(variables) %>%
      pull
    message("\n* Detected diagnose many unique value")
    message(paste(" -", vars, collapse = "\n"))
  }

  suppressMessages(
    df <- missing_level <- .data %>%
      compare_target_category(add_character = add_character)
  )

  if (!is.null(df)) {
    missing_level <- df %>%
      group_by(variable) %>%
      summarise(n_levels = n(),
        train_missing_nlevel = sum(is.na(train)),
        test_missing_nlevel = sum(is.na(test))) %>%
      filter(train_missing_nlevel > 0 | test_missing_nlevel > 0)

    names(missing_level)[1] <- "variables"

    if (nrow(missing_level)) {
      vars <- missing_level %>%
        select(variables) %>%
        pull
      message("\n* Detected diagnose missing levels")
      message(paste(" -", vars, collapse = "\n"))
    }
  } else{
    missing_level <- tibble::tribble(
      ~variables, ~n_levels, ~train_missing_nlevel, ~test_missing_nlevel,
      "a", 1L, 1L, 1L)
    missing_level <- missing_level[-1, ]
  }

  list(missing_value = missing_value, single_value = single_value,
    uniq_rate = uniq_rate, missing_level = missing_level)
}


#' @rdname cleanse.data.frame
#' @export
cleanse <- function(.data, ...) {
  UseMethod("cleanse", .data)
}


#' Cleansing the dataset for classification modeling
#'
#' @description Diagnosis of similarity between datasets splitted by train set and set included in the "split_df" class. and cleansing the "split_df" class
#' @param .data an object of class "split_df", usually, a result of a call to split_df().
#' @param add_character logical. Decide whether to include text variables in the
#' compare of categorical data. The default value is FALSE, which also not includes character variables.
#' @param uniq_thres numeric. Set a threshold to removing variables when the ratio of unique values(number of unique values / number of observation) is greater than the set value.
#' @param missing logical. Set whether to removing variables including missing value
#' @param ... further arguments passed to or from other methods.
#' @details
#' Remove the detected variables from the diagnosis using the compare_diag() function.
#'
#' @return An object of class "split_df".
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
#' sb %>%
#'   cleanse
#'
#' @method cleanse split_df
#' @export
cleanse.split_df <- function(.data, add_character = FALSE, uniq_thres = 0.9,
  missing = FALSE, ...) {
  diag <- .data %>%
    compare_diag(add_character = add_character, uniq_thres = uniq_thres,
      miss_msg = missing)

  if (!missing) {
    diag <- diag[-1]
  }

  vars <- unlist(sapply(diag, function(x) x[, "variables"]))

  if (length(vars) >= 1) {
    .data %>%
      dplyr::select(-vars)
  } else {
    message("There were no diagnostics issues")
    .data
  }
}


#' Extract train/test dataset
#'
#' @description Extract train set or test set from split_df class object
#'
#' @param x an object of class "split_df", usually, a result of a call to split_df().
#' @param set character. Specifies whether the extracted data is a train set or a test set.
#' You can use "train" or "test".
#'
#' @details
#' Extract the train or test sets based on the parameters you defined when creating split_df with split_by().
#'
#' @return an object of class "tbl_df".
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
#' train <- sb %>%
#'   extract_set(set = "train")
#'
#' test <- sb %>%
#'   extract_set(set = "test")
#'
#' @import dplyr
#' @export
extract_set <- function(x, set = c("train", "test")) {
  if(is(x) != "split_df") {
    stop("x is not split_df class")
  }

  set <- match.arg(set)

  x %>%
    filter(split_flag == set) %>%
    ungroup() %>%
    dplyr::select(-split_flag)
}



