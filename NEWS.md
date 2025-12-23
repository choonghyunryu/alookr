# alookr 0.5.0

## MINOR CHANGES
  
* Fix logic in `predictor()` and `classifier_xgboost` that is realted new xgboost package. 
    - (#10)
    
    
    
# alookr 0.4.0

## MINOR CHANGES
  
* Fix logic in `compare_plot()` that is "using ggmosaic package.". 
    - (#9)
    
    
    
# alookr 0.3.91

## MINOR CHANGES

* fixed Rd file(s) with Rd \link{} targets missing package #8
    - cleanse.data.frame.Rd: tbl_df
    - split_by.data.frame.Rd: tbl_df
    - treatment_corr.Rd: tbl_df



# alookr 0.3.9

## MINOR CHANGES
  
* Fix error in `treatment_corr()` that is "All columns in a tibble must be vectors." error. 
    - (#6, thanks to Cathy Tomson)
    
    
    
# alookr 0.3.8

## BUG FIXES
  
* Fix error in `treatment_corr()` that is "All columns in a tibble must be vectors." error. 
    - (#6, thanks to Cathy Tomson)



# alookr 0.3.7      
      
## MAJOR CHANGES
  
* Removed `plan(multiprocess)` from logic for parallel processing. 
    - Because, `plan(multiprocess)` of future is deprecated. (#2, thanks to Henrik Bengtsson)
      
## MINOR CHANGES
  
* Remove the warning of "UNRELIABLE VALUE" with `seed = TRUE` in future function. 
      
## BUG FIXES
  
* Fix error in `run_performance()` that is "replacement has length zero" error.
    - (#5, thanks to Muhammad Fawad)
      
      
      
# alookr 0.3.6
      
## MINOR CHANGES
  
* Implemented a function to replace the unbalanced package used in the process of performing split data. 
    - This is because unbalanced packages have been removed from CRAN. (#3)
      
      

# alookr 0.3.5
      
## BUG FIXES
  
* Fix error in glmnet when `run_predict()` is performed with test data that has more variables than train data.
    
    
    
# alookr 0.3.4
      
## MAJOR CHANGES
  
* add xgboosting methodlogy for binary classifier.
* add lasso regression model for binary classifier.      
      
     
 
# alookr 0.3.3
      
## BUG FIXES
  
* `run_predict()` fixed error when try to predict on dataset without the response variable 
    - (thanks @shivakhanal, #1).
    
## MINOR CHANGES
  
* `run_models()`, `run_predict()`, `run_performance()` not support future::multiprocess when running R from RStudio. 



# alookr 0.3.2
      
## BUG FIXES

* Fixed explanation errors in `Classification Modeling` vignettes for debian linux.
    
## MINOR CHANGES

* Renamed `compare_category()` to `compare_target_category()`. 
    - This is because it overlaps the function name of the dlookr package.
* Renamed `compare_numeric()` to `compare_target_numeric()`. 
    - This is because it overlaps the function name of the dlookr package.
* `compare_target_category()` modified from `is.tibble()`, `as.tibble()` to `is_tibble()`, `as_tibble()`.  
* `compare_diag()` modified from is.tibble(), as.tibble() to is_tibble(), as_tibble().  
* `sampling_target()` modified from as.tbl() to tibble::as_tibble().
    
    

# alookr 0.3.1
      
## BUG FIXES

* Fixed explanation errors in `Cleansing the dataset` vignettes.
* Fixed explanation errors in `Classification Modeling` vignettes.
* Modified explanation errors in `Splitting the dataset` vignettes.
    