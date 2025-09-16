#' Cleansing the dataset for classification modeling
#'
#' @description The cleanse() cleanse the dataset for classification modeling
#' @param .data a data.frame or a \code{\link[tibble]{tbl_df}}.
#' @param uniq logical. Set whether to remove the variables whose unique value is one.
#' @param uniq_thres numeric. Set a threshold to removing variables when the ratio of unique values(number of unique values / number of observation) is greater than the set value.
#' @param char logical. Set the change the character to factor.
#' @param missing logical. Set whether to removing variables including missing value
#' @param verbose logical. Set whether to echo information to the console at runtime.
#' @param ... further arguments passed to or from other methods.
#' @details
#' This function is useful when fit the classification model.
#' This function does the following.:
#' Remove the variable with only one value. And remove variables that have a unique number of values relative to the number of observations for a character or categorical variable. In this case, it is a variable that corresponds to an identifier or an identifier. And converts the character to factor.
#'
#' @return An object of data.frame or train_df. and return value is an object of the same type as the .data argument.
#' @examples
#' # create sample dataset
#' set.seed(123L)
#' id <- sapply(1:1000, function(x)
#'   paste(c(sample(letters, 5), x), collapse = ""))
#'
#' year <- "2018"
#'
#' set.seed(123L)
#' count <- sample(1:10, size = 1000, replace = TRUE)
#'
#' set.seed(123L)
#' alpha <- sample(letters, size = 1000, replace = TRUE)
#'
#' set.seed(123L)
#' flag <- sample(c("Y", "N"), size = 1000, prob = c(0.1, 0.9), replace = TRUE)
#'
#' dat <- data.frame(id, year, count, alpha, flag, stringsAsFactors = FALSE)
#' # structure of dataset
#' str(dat)
#'
#' # cleansing dataset
#' newDat <- cleanse(dat)
#'
#' # structure of cleansing dataset
#' str(newDat)
#'
#' # cleansing dataset
#' newDat <- cleanse(dat, uniq = FALSE)
#'
#' # structure of cleansing dataset
#' str(newDat)
#'
#' # cleansing dataset
#' newDat <- cleanse(dat, uniq_thres = 0.3)
#'
#' # structure of cleansing dataset
#' str(newDat)
#'
#' # cleansing dataset
#' newDat <- cleanse(dat, char = FALSE)
#'
#' # structure of cleansing dataset
#' str(newDat)
#'
#' @method cleanse data.frame
#' @importFrom dlookr diagnose
#' @import dplyr
#' @import cli
#' @export
cleanse.data.frame <- function(.data, uniq = TRUE, uniq_thres = 0.1, char = TRUE,
  missing = FALSE, verbose = TRUE, ...) {
  if (missing) {
    if (verbose) {
      cli::cat_rule(
        left = "Checking missing value",
        right = "included NA",
        col = "cyan",
        width = 75
      )
    }

    vars_na <- dlookr::diagnose(.data) %>%
      filter(missing_count > 0) %>%
      dplyr::select(variables) %>%
      mutate(variables = as.character(variables)) %>%
      pull

    if (length(vars_na) > 0) {
      if (verbose) {
        message("remove variables whose included NA")
        cli::cat_bullet(vars_na, bullet_col = "red", col = "red")
        cat("\n")
      }

      .data <- .data %>%
        dplyr::select(-vars_na)
    } else {
      if (verbose) {
        cat(cli::col_grey("No variables with missing values.\n\n"))
      }
    }
  }

  if (uniq) {
    if (verbose) {
      cli::cat_rule(
        left = "Checking unique value",
        right = "unique value is one",
        col = "cyan",
        width = 75
      )
    }

    vars_uniq <- .data %>%
      dlookr::diagnose() %>%
      filter(unique_count == 1) %>%
      dplyr::select(variables) %>%
      pull

    if (length(vars_uniq) > 0) {
      if (verbose) {
        message("remove variables that unique value is one")
        cli::cat_bullet(vars_uniq, bullet_col = "red", col = "red")
        cat("\n")
      }

      .data <- .data %>%
        select(-vars_uniq)
    } else {
      if (verbose) {
        cat(cli::col_grey("No variables that unique value is one.\n\n"))
      }
    }

    if (verbose) {
      cli::cat_rule(
        left = "Checking unique rate",
        right = "high unique rate",
        col = "cyan",
        width = 75
      )
    }

    vars_rate <- .data %>%
      dlookr::diagnose() %>%
      filter(types %in% c("character", "factor", "ordered")) %>%
      filter(unique_rate >= uniq_thres) %>%
      dplyr::select(variables, unique_count, unique_rate)

    if (nrow(vars_rate) > 0) {
      if (verbose) {
        message("remove variables with high unique rate")
        cli::cat_bullet(sprintf("%s = %s(%s)\n", vars_rate$variables,
                                vars_rate$unique_count, vars_rate$unique_rate),
                        bullet_col = "red", col = "red")
      }

      vars_rate <- vars_rate %>%
        dplyr::select(variables) %>%
        pull

      .data <- .data %>%
        dplyr::select(-vars_rate)
    } else {
      if (verbose) {
        cat(cli::col_grey("No variables that high unique rate.\n\n"))
      }
    }
  }

  if (char) {
    if (verbose) {
      cli::cat_rule(
        left = "Checking character variables",
        right = "categorical data",
        col = "cyan",
        width = 75
      )
    }

    vars_char <- dlookr::get_class(.data) %>%
      filter(class == "character") %>%
      dplyr::select(variable) %>%
      mutate(variable = as.character(variable)) %>%
      pull

    if (length(vars_char) > 0) {
      if (verbose) {
        message("converts character variables to factor")
        cli::cat_bullet(vars_char, bullet_col = "red", col = "red")
        cat("\n")
      }

      .data <- .data %>%
        mutate_if(is.character, factor)
    } else {
      if (verbose) {
        cat(cli::col_grey("No character variables.\n\n"))
      }
    }
  }

  .data
}



#' Diagnosis and removal of highly correlated variables
#'
#' @description The treatment_corr() diagnose pairs of highly correlated variables or remove on of them.
#' @param .data a data.frame or a \code{\link[tibble]{tbl_df}}.
#' @param corr_thres numeric. Set a threshold to detecting variables when correlation greater then threshold.
#' @param treat logical. Set whether to removing variables
#' @param verbose logical. Set whether to echo information to the console at runtime.
#' @details The correlation coefficient of pearson is obtained for continuous variables and the correlation coefficient of spearman for categorical variables.
#'
#' @return An object of data.frame or train_df. and return value is an object of the same type as the .data argument. However, several variables can be excluded by correlation between variables.
#' @examples
#' # numerical variable
#' x1 <- 1:100
#' set.seed(12L)
#' x2 <- sample(1:3, size = 100, replace = TRUE) * x1 + rnorm(1)
#' set.seed(1234L)
#' x3 <- sample(1:2, size = 100, replace = TRUE) * x1 + rnorm(1)
#'
#' # categorical variable
#' x4 <- factor(rep(letters[1:20], time = 5))
#' set.seed(100L)
#' x5 <- factor(rep(letters[1:20 + sample(1:6, size = 20, replace = TRUE)], time = 5))
#' set.seed(200L)
#' x6 <- factor(rep(letters[1:20 + sample(1:3, size = 20, replace = TRUE)], time = 5))
#' set.seed(300L)
#' x7 <- factor(sample(letters[1:5], size = 100, replace = TRUE))
#'
#' exam <- data.frame(x1, x2, x3, x4, x5, x6, x7)
#' str(exam)
#' head(exam)
#'
#' # default case
#' treatment_corr(exam)
#'
#' # not removing variables
#' treatment_corr(exam, treat = FALSE)
#'
#' # Set a threshold to detecting variables when correlation greater then 0.9
#' treatment_corr(exam, corr_thres = 0.9, treat = FALSE)
#'
#' # not verbose mode
#' treatment_corr(exam, verbose = FALSE)
#'
#' @importFrom dlookr diagnose
#' @importFrom tibble as_tibble add_column
#' @importFrom stats cor
#' @import dplyr
#' @export
treatment_corr <- function(.data, corr_thres = 0.8, treat = TRUE, verbose = TRUE) {
  ## Pearson correlation for numerical variables
  n_numeric <- .data %>%
    diagnose() %>%
    filter(types %in% c("integer", "numeric")) %>%
    filter(!variables %in% "TARGET") %>%
    select(variables) %>%
    pull() %>% 
    length()
  
  if (n_numeric > 2) {
    corr <- .data %>%
      dlookr::correlate() %>%
      filter(abs(coef_corr) > corr_thres) %>%
      filter(as.integer(var1) > as.integer(var2))
    
    vars <- corr %>%
      distinct(var2) %>%
      pull %>%
      as.character
    
    if (nrow(corr) > 0) {
      if (verbose) {
        message(sprintf("* remove variables whose strong correlation (pearson >= %s)",
                        corr_thres))
        message(paste(" - remove ", format(corr$var2), " : with ", corr$var1,
                      " (", round(corr$coef_corr, 4), ")\n", sep = ""))
      }
      
      if (treat) {
        .data <- .data %>%
          dplyr::select(-vars)
      }
    }    
    
    n_corr <- nrow(corr)
  } else {
    n_corr <- 0
  }  

  ## Spearman correlation for categorical variables
  vars <- .data %>%
    diagnose() %>%
    filter(types %in% "factor") %>%
    filter(!variables %in% "TARGET") %>%
    select(variables) %>%
    pull

  if (length(vars) > 2) {
    M <- .data %>%
      select(vars) %>%
      mutate_all(as.integer) %>%
      cor(method = "spearman")
    
    m <- as.vector(M)
    tab <- tibble::as_tibble(expand.grid(var1 = row.names(M),
                                         var2 = row.names(M)))
    corr2 <- tibble::add_column(tab, coef_corr = m) %>%
      filter(var1 != var2) %>%
      filter(var1 %in% vars) %>%
      filter(abs(coef_corr) > corr_thres) %>%
      filter(as.integer(var1) > as.integer(var2))
    
    vars <- corr2 %>%
      distinct(var2) %>%
      pull %>%
      as.character
    
    if (nrow(corr2) > 0) {
      if (verbose) {
        message(sprintf("* remove variables whose strong correlation (spearman >= %s)",
                        corr_thres))
        message(paste(" - remove ", format(corr2$var2), " : with ", corr2$var1,
                      " (", round(corr2$coef_corr, 4), ")\n", sep = ""))
      }
      
      if (treat) {
        .data <- .data %>%
          dplyr::select(-vars)
      }
    }  
    
    n_corr2 <- nrow(corr2)
  } else {
    n_corr2 <- 0
  }  

  if ((n_corr + n_corr2) == 0 & verbose) {
    message("All correlation coefficient is below threshold")
  }  
  
  if (treat) {
    .data
  }
}


trans_matrix <- function(.data) {
  data_type <- dlookr::get_class(.data) %>%
    dplyr::select(class) %>%
    pull

  idx_character <- data_type %in% "character"
  idx_factor <- data_type %in% c("factor", "ordered")

  if (sum(idx_character) > 1) {
    .data <- .data %>%
      transmute_if(idx_character, function(x) as.numeric(as.factor(x)))
  }

  if (sum(idx_factor) > 1) {
    .data <- .data %>%
      transmute_if(idx_factor, as.numeric)
  }

  as.matrix(.data)
}

