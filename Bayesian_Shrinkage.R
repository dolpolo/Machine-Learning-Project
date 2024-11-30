# ---- SET DIRECTORY
getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)


# ---- LIBRARIES
library(dplyr)
library(tidyverse)
library(tseries)
library(ggplot2)
library(writexl)
library(readxl)
library(lubridate)
library(pls)
library(lars)
library(RSpectra)
library(glmnet)
library(xtable)
library(here)




# **************************************************************************** #
# ********************** BAYEASIAN SHRINKAGE PREDICTIONS ********************* #
# **************************************************************************** #  

# This code aims to replicate the analysis by Christine De Mol, Domenico Giannone and Lucrezia Reichlin
# using the paper Forecasting using a large number of predictors: Is Bayesian shrinkage a valid
# alternative to principal components?" on a different dataset on EA countries

# the code compares the performances of
# RIDGE regression
# LASSO regression
# PC regression

################################################################################

# ---- LOAD DATA ----
# load EA_data until 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

# ---- CALL FUNCTIONS ----
# call the Bayesian Shrinkage functions
source("R/functions/Bayesian_shrinkage_functions.R")

# ---- SET PARAMETERS ----
# Dependent variables to be predicted
nn <- c(1,37,97)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 25, 40, 50)  # Number of principal components
K <- rr  # Number of predictors with LASSO
INfit <- seq(0.1, 0.9, by = 0.1)  # In-sample residual variance explained by Ridge
HH <- c(4)  # Steap-ahead prediction
Jwind <- 56  # Rolling window

# ********************************* IN SAMPLE ******************************** #

# Start dates of the out-of-sample evaluation
start_y <- 2014
start_m <- 01

DATA <- as.matrix(EAdataQ[, -1])  # Matrix of data: time in rows, variables in columns
series <- colnames(EAdataQ)
X <- DATA

# target variables
target_var <- colnames(X)[nn]

# Panel dimensions
TT <- nrow(X)
NN <- ncol(X)

# Find the index of the start of the out-of-sample
start_sample <- which(year(EAdataQ$Time) == start_y & month(EAdataQ$Time) == start_m)
if (Jwind > start_sample) stop("The rolling window cannot be larger than the first evaluation sample")

j0 <- start_sample - Jwind + 1
x <- X[j0:start_sample, ]


# ==============================================================================
# ============================= RIDGE PENALIZATION ============================= 
# ==============================================================================

# Setting the RIDGE penalty parameter (requires SET_ridge function)
nu_ridge <- list()
for (jfit in seq_along(INfit)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      nu_ridge[[paste(h, k, sep = "_")]][jfit] <- SET_ridge(x[, nn[k]], x, p, INfit[jfit], h)
    }
  }
}
nu_ridge


# ==============================================================================
# ============================= LASSO PENALIZATION ============================= 
# ==============================================================================

# Setting the LASSO penalty parameter (requires SET_lasso function)
nu_LASSO <- list()
for (jK in seq_along(K)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      nu_LASSO[[paste(h, k, sep = "_")]][jK] <- SET_lasso(x[, nn[k]], x, p, K[jK], h)
    }
  }
}
nu_LASSO


# Assigning dependent variable names
for (k in seq_along(nn)){
  # Construct the name dynamically
  element_name <- paste0("4_", nn[k])
  
  # Assign the name to the i-th element of nu_ridge
  names(nu_ridge)[k] <- element_name
}


# Assigning INfit values to each elememnt of the list
for (element_name in names(nu_ridge)) {
  
  # Assign names to each value in `nu_ridge[[element_name]]` based on `residual_variances`
  names(nu_ridge[[element_name]]) <- INfit
}

# ---- RESCALING NU (N*T) ----

T <- nrow(x)
N <- ncol(x)
multiply <- as.numeric(T*N)

# Multiply each element of nu_ridge by T * N using lapply
NT_nu_ridge <- lapply(nu_ridge, function(x) as.numeric(x) * multiply)
NT_nu_ridge

NT_nu_ridge <- as.data.frame(NT_nu_ridge)
NT_nu_ridge_t <- t(NT_nu_ridge)

# Transpose the matrix
NT_nu_ridge_t <- t(NT_nu_ridge)

# Convert to LaTeX
latex_output <- xtable(NT_nu_ridge_t)

# Print LaTeX code
print(latex_output, type = "latex")

target_var_names <- as.character(target_var)

# Combine the names to rename the columns
col_names <- c(as.character(INfit))

# Apply new names to columns of dataframe
colnames(NT_nu_ridge_t) <- col_names
rownames(NT_nu_ridge_t) <- target_var_names 

# LaTeX
LTX_NT_nu_ridge_t <- xtable(NT_nu_ridge_t)

print(LTX_NT_nu_ridge_t, type = "latex")


# Assigning dependent variable names
for (k in seq_along(nn)){
  # Construct the name dynamically
  element_name <- paste0("4_", nn[k])
  
  # Assign the name to the i-th element of nu_ridge
  names(nu_LASSO)[k] <- element_name
}


# Assigning INfit values to each elememnt of the list
for (element_name in names(nu_LASSO)) {
  
  # Assign names to each value in `nu_ridge[[element_name]]` based on `residual_variances`
  names(nu_LASSO[[element_name]]) <- K
}



# ****************************** OUT OF SAMPLE ******************************* #

# ==============================================================================
# =============================== RIDGE PREDICTION ============================= 
# ==============================================================================

# Define lengths for each level
outer_length <- TT
mid_length <- length(nn)
inner_length <- length(K)

# Initialize each list using the function
pred_br <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RIDGE <- initialize_nested_list(outer_length, c(mid_length, inner_length))
true_r <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RW_r <- initialize_nested_list(outer_length, c(mid_length, inner_length))

# Perform the out-of-sample forecasting exercise using LASSO
for (j in start_sample:(TT - HH)) {
  
  # Define sample 
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Data available for every point in time 
  x <- X[j0:j, ]  
  
  for (k in seq_along(nn)) {
    
    for (jfit in seq_along(INfit)) {
      
      # Normalization constants 
      const <- 4
      
      # Compute LASSO forecasts 
      for (h in HH) {  # Loop on steps ahead 
        pred_br[[j]][[k]][[jfit]]<- RIDGE_pred(x[, nn[k]], x, p, nu_ridge[[k]][[jfit]], h)
        RIDGE[[j+h]][[k]][[jfit]] <- pred_br[[j]][[k]][[jfit]][["pred"]] * const
        
        
        # Compute true value to be forecasted
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_r[[j+h]][[k]][[jfit]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_r[[j+h]][[k]][[jfit]] <- temp * const
        
      }
    }
  }
}



################################################################################

        
# ==============================================================================
# =============================== LASSO PREDICTION ============================= 
# ==============================================================================

# Define lengths for each level
outer_length <- TT
mid_length <- length(nn)
inner_length <- length(K)

# Initialize each list using the function
pred_bl <- initialize_nested_list(outer_length, c(mid_length, inner_length))
LASSO <- initialize_nested_list(outer_length, c(mid_length, inner_length))
true_l <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RW_l <- initialize_nested_list(outer_length, c(mid_length, inner_length))


# Perform the out-of-sample forecasting exercise using LASSO
for (j in start_sample:(TT - HH)) {
  
  # Define sample 
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  

  x <- X[j0:j, ]  # Data available at every point in time 
  
  for (k in seq_along(nn)) {
    
    for (jK in seq_along(K)) {
      
      const <- 4
      
      # LASSO forecasts
      for (h in HH) {  
        pred_bl[[j]][[k]][[jK]]<- LASSO_pred(x[, nn[k]], x, p, nu_LASSO[[k]][[jK]], h)
        LASSO[[j+h]][[k]][[jK]] <- pred_bl[[j]][[k]][[jK]][["pred"]] * const
        
        # Compute true value to be forecasted 
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_l[[j+h]][[k]][[jK]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_l[[j+h]][[k]][[jK]] <- temp * const
        
      }
    }
  }
}




# ==============================================================================
# ====================== PRINCIPAL COMPONENT PREDICTION ======================== 
# ==============================================================================


# Define lengths for each level
outer_length <- TT
mid_length <- length(nn)
inner_length <- length(K)

# Initialize each list using the function
pred_bpc <- initialize_nested_list(outer_length, c(mid_length, inner_length))
PC <- initialize_nested_list(outer_length, c(mid_length, inner_length))
true_pc <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RW_pc <- initialize_nested_list(outer_length, c(mid_length, inner_length))


# Perform the out-of-sample forecasting exercise using PCR
for (j in start_sample:(TT - HH)) {
  
  # Define sample
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  
  x <- X[j0:j, ] 
  
  for (k in seq_along(nn)) {
    
    for (jr in seq_along(rr)) {
      
     
      const <- 4
      
      # Compute PC forecasts
      for (h in HH) {  
        pred_bpc[[j]][[k]][[jr]]<- PC_pred(x[, nn[k]], x, p, rr[jr], h)
        PC[[j+h]][[k]][[jr]] <- pred_bpc[[j]][[k]][[jr]][["pred"]] * const
        
        
        
        # Compute true value to be forecasted 
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_pc[[j+h]][[k]][[jr]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_pc[[j+h]][[k]][[jr]] <- temp * const
        
      }
    }
  }
}

################################################################################


# ============================== MODEL COMPARISON ==============================


        
# ****************************** EVALUATION SAMPLE **************************** #

# Dates of the beginning of the evaluation sample
first_y <- 2015
first_m <- 1

# Find the index for the first out-of-sample period
ind_first <- which(year(EAdataQ$Time) == first_y & month(EAdataQ$Time) == first_m)

# Dates of the end of the evaluation sample
last_y <- 2019
last_m <- 10

# Find the index for the last out-of-sample period
ind_last <- which(year(EAdataQ$Time) == last_y & month(EAdataQ$Time) == last_m)


# ************************ RIDGE MSFE AND VISUALIZATION ************************


# Turn the prediction for each variable into a matrix

# ---- MSFE computation

# Number of variables and length of intervals
variable_count <- length(nn)
interval_length <- length(ind_first:length(RIDGE))

# Initialize output lists
RIDGE_output <- initialize_output_list(variable_count, interval_length)
true_r_output <- initialize_output_list(variable_count, interval_length)
RW_r_output <- initialize_output_list(variable_count, interval_length)

# Process PC predictions
RIDGE_output <- process_output(RIDGE, RIDGE_output, ind_first, variable_count)

# Process true values
true_r_output <- process_output(true_r, true_r_output, ind_first, variable_count)

# Process Random Walk predictions
RW_r_output <- process_output(RW_r, RW_r_output, ind_first, variable_count)

# Calculate squared differences
diff_RIDGE_output <- calculate_squared_differences(true_r_output, RIDGE_output,
                                                variable_count, interval_length)
diff_RW_r_output <- calculate_squared_differences(true_r_output, RW_r_output, variable_count,
                                                   interval_length)

# Calculate mean squared forecast error (MSFE)
MSFE_RIDGE <- calculate_msfe(diff_RIDGE_output, variable_count)
MSFE_RW_r <- calculate_msfe(diff_RW_r_output, variable_count)

# Display results
print(MSFE_RIDGE)
print(MSFE_RW_r)

# ---- MFSE Ratio

# Convert MSFE_FS and MSFE_RW lists to matrices
MSFE_RIDGE_matrix <- convert_to_named_matrix_r(MSFE_RIDGE, target_var, INfit)
print(MSFE_RIDGE_matrix)

MSFE_RW_r_matrix <- convert_to_named_matrix(MSFE_RW_r, target_var, INfit)
print(MSFE_RW_r_matrix)

# Calculate and display the ratio of MSFE_FS to MSFE_RW
MSFE_RIDGE_ratio <- MSFE_RIDGE_matrix / MSFE_RW_r_matrix
print(MSFE_RIDGE_ratio)


# ---- Visualization

# Convert the dataframe to long format, including an identifier for the rows
MSFE_RIDGE_matrix_long <- MSFE_RIDGE_matrix %>%
  rownames_to_column(var = "Variable") %>%
  pivot_longer(cols = -Variable, names_to = "Penalization", values_to = "Value")

ggplot(MSFE_RIDGE_matrix_long, aes(x = Penalization, y = Value, color = Variable, group = Variable)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  labs(
    title = "MSFE RIDGE",
    x = "In-sample residual variances",
    y = "MSFE"
  ) +
  facet_wrap(~ Variable, scales = "free_y") + 
  scale_color_brewer(palette = "Set1") +
  theme_minimal(base_size = 14) + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  
    axis.title = element_text(face = "bold"),
    legend.position = "bottom", 
    legend.title = element_blank(), 
    strip.text = element_text(face = "bold", size = 12)
  )




# ************************ LASSO MSFE AND VISUALIZATION ************************

# Turn the prediction for each variable into a matrix

# ---- MSFE computation

# Number of variables and length of intervals
variable_count <- length(nn)
interval_length <- length(ind_first:length(LASSO))

# Initialize output lists
LASSO_output <- initialize_output_list(variable_count, interval_length)
true_l_output <- initialize_output_list(variable_count, interval_length)
RW_output <- initialize_output_list(variable_count, interval_length)

# Process LASSO predictions
LASSO_output <- process_output(LASSO, LASSO_output, ind_first, variable_count)

# Process true values
true_l_output <- process_output(true_l, true_l_output, ind_first, variable_count)

# Process Random Walk predictions
RW_output <- process_output(RW_l, RW_output, ind_first, variable_count)

# Calculate squared differences
diff_l_output <- calculate_squared_differences(true_l_output, LASSO_output,
                                                variable_count, interval_length)
diff_RW_output <- calculate_squared_differences(true_l_output, RW_output, variable_count,
                                                interval_length)

# Calculate mean squared forecast error (MSFE)
MSFE_l <- calculate_msfe(diff_l_output, variable_count)
MSFE_RW <- calculate_msfe(diff_RW_output, variable_count)

# Display results
print(MSFE_l)
print(MSFE_RW)

# ---- MSFE Ratio

# Convert MSFE_l and MSFE_RW lists to matrices
MSFE_l_matrix <- convert_to_named_matrix(MSFE_l, target_var, K)
print(MSFE_l_matrix)

MSFE_RW_matrix <- convert_to_named_matrix(MSFE_RW, target_var, K)
print(MSFE_RW_matrix)

# Calculate and display the ratio of MSFE_l to MSFE_RW
MSFE_l_ratio <- MSFE_l_matrix / MSFE_RW_matrix
print(MSFE_l_ratio)


# ---- Visualization

# Convert the dataframe to long format, including an identifier for the rows
MSFE_l_matrix_long <- MSFE_l_matrix %>%
  rownames_to_column(var = "Variable") %>%
  pivot_longer(cols = -Variable, names_to = "Penalization", values_to = "Value")

MSFE_l_matrix_long <- MSFE_l_matrix_long %>%
  mutate(Penalization = factor(Penalization, levels = unique(Penalization)))


ggplot(MSFE_l_matrix_long, aes(x = Penalization, y = Value, color = Variable, group = Variable)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  labs(
    title = "MSFE LASSO",
    x = "Number of predictors with non-zero coefficient",
    y = "MSFE"
  ) +
  facet_wrap(~ Variable, scales = "free_y") + 
  scale_color_brewer(palette = "Set1") +
  theme_minimal(base_size = 14) + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  
    axis.title = element_text(face = "bold"),
    legend.position = "bottom", 
    legend.title = element_blank(), 
    strip.text = element_text(face = "bold", size = 12)
  )


# *************************** PC MSFE AND VISUALIZATION ************************


# Turn the prediction for each variable into a matrix

# ---- MSFE computation

# Number of variables and length of intervals
variable_count <- length(nn)
interval_length <- length(ind_first:length(PC))

# Initialize output lists
PC_output <- initialize_output_list(variable_count, interval_length)
true_pc_output <- initialize_output_list(variable_count, interval_length)
RW_pc_output <- initialize_output_list(variable_count, interval_length)

# Process PC predictions
PC_output <- process_output(PC, PC_output, ind_first, variable_count)

# Process true values
true_pc_output <- process_output(true_pc, true_pc_output, ind_first, variable_count)

# Process Random Walk predictions
RW_pc_output <- process_output(RW_pc, RW_pc_output, ind_first, variable_count)

# Calculate squared differences
diff_PC_output <- calculate_squared_differences(true_pc_output, PC_output,
                                                variable_count, interval_length)
diff_RW_pc_output <- calculate_squared_differences(true_pc_output, RW_pc_output, variable_count,
                                                interval_length)

# Calculate mean squared forecast error (MSFE)
MSFE_PC <- calculate_msfe(diff_PC_output, variable_count)
MSFE_RW_pc <- calculate_msfe(diff_RW_pc_output, variable_count)

# Display results
print(MSFE_PC)
print(MSFE_RW_pc)

# ---- MFSE Ratio

# Convert MSFE_FS and MSFE_RW lists to matrices
MSFE_PC_matrix <- convert_to_named_matrix(MSFE_PC, target_var, K)
print(MSFE_PC_matrix)

MSFE_RW_pc_matrix <- convert_to_named_matrix(MSFE_RW_pc, target_var, K)
print(MSFE_RW_pc_matrix)

# Calculate and display the ratio of MSFE_FS to MSFE_RW
MSFE_PC_ratio <- MSFE_PC_matrix / MSFE_RW_pc_matrix
print(MSFE_PC_ratio)

getwd()
setwd("C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project")
saveRDS(MSFE_PC_matrix, file= "Results/MSFE/MSFE_PC_matrix.rds")
saveRDS(MSFE_PC_ratio, file= "Results/MSFE/MSFE_PC_ratio.rds")

# ---- Visualization

# Convert the dataframe to long format, including an identifier for the rows
MSFE_PC_matrix_long <- MSFE_PC_matrix %>%
  rownames_to_column(var = "Variable") %>%
  pivot_longer(cols = -Variable, names_to = "Penalization", values_to = "Value")

MSFE_PC_matrix_long <- MSFE_PC_matrix_long %>%
  mutate(Penalization = factor(Penalization, levels = unique(Penalization)))

ggplot(MSFE_PC_matrix_long, aes(x = Penalization, y = Value, color = Variable, group = Variable)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  labs(
    title = "MSFE PC",
    x = "Number of Principal Components",
    y = "MSFE"
  ) +
  facet_wrap(~ Variable, scales = "free_y") + 
  scale_color_brewer(palette = "Set1") +
  theme_minimal(base_size = 14) + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  
    axis.title = element_text(face = "bold"),
    legend.position = "bottom", 
    legend.title = element_blank(), 
    strip.text = element_text(face = "bold", size = 12)
  )




# ==============================================================================
# =================================== BEST MODEL ===============================
# ==============================================================================


# ---- RIDGE ----

# Initialize an empty list to save the results
best_r_model <- data.frame(Variable = rownames(MSFE_RIDGE_matrix),
                           Best_MSFE = numeric(nrow(MSFE_RIDGE_matrix)),
                           Best_Penalization = character(nrow(MSFE_RIDGE_matrix)),
                           nu_RIDGE_r = numeric(nrow(MSFE_RIDGE_matrix)),
                           stringsAsFactors = FALSE)

# Create a mapping between the names of the penalties and the numerical indices
penalization_map <- c( "0.1" = 1, "0.2" = 2, "0.3" = 3, "0.4" = 4, 
                       "0.5" = 5, "0.6" = 6, "0.7" = 7, "0.8" = 8, "0.9" = 9 )

# Loop for every row of matrix MSFE
for (i in 1:nrow(MSFE_RIDGE_matrix)) {
  
  # Find the index of the column with the minimum value in row i
  min_index <- which.min(MSFE_RIDGE_matrix[i, ])
  
  #Assign the minimum value and the column name to the results list
  best_r_model$Best_MSFE[i] <- MSFE_RIDGE_matrix[i, min_index]
  best_r_model$Best_Penalization[i] <- colnames(MSFE_RIDGE_matrix)[min_index]
  
  #  Get the name of the best penalty for this variable
  best_penalization <- best_r_model$Best_Penalization[i]
  
  # Check if the penalty is present in the mapping
  if (best_penalization %in% names(penalization_map)) {
    
    # Get the index corresponding to the penalty
    penalization_index <- penalization_map[best_penalization]
    
    # Select the corresponding prediction from nu_ridge
    best_r_model$nu_RIDGE_r[i] <- nu_ridge[[i]][[penalization_index]]
    
  } else {
    # Handle the case where the penalty is not found
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_RIDGE_matrix)[i]))
    best_r_model$nu_RIDGE_r[i] <- NA  # Assign NA if penalty not found
  }
}

# Visualize results
print(best_r_model)

# Initialize an empty matrix for the best predictions for each year.
best_r_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(RIDGE))), ncol = nrow(MSFE_RIDGE_matrix))
rownames(best_r_prediction) <- seq(from = ind_first, to = length(RIDGE))  
colnames(best_r_prediction) <- rownames(MSFE_RIDGE_matrix) 

# Loop for every variable (row of MSFE_RIDGE_matrix)
for (i in 1:nrow(MSFE_RIDGE_matrix)) {
  
  #Find the index of the column with the minimum value in row i (the optimal penalty)
  min_index <- which.min(MSFE_RIDGE_matrix[i, ])
  
  # Get the name of the best penalty for this variable
  best_penalization <- colnames(MSFE_RIDGE_matrix)[min_index]
  
  # Get the index of the corresponding penalty
  penalization_index <- penalization_map[best_penalization]
  
  # For each year (from ind_first to RIDGE), select the prediction corresponding to the optimal penalty
  for (j in ind_first:length(RIDGE)) {  
    # Extract the list of predictions for variable i and year j from RIDGE
    pred_blist_for_year_var <- RIDGE[[j]][[i]]
    
    # Check if penalization_index is within the bounds of the predictions list
    if (penalization_index <= length(pred_blist_for_year_var)) {
      # Save the prediction corresponding to the best penalty parameter
      best_r_prediction[j - ind_first + 1, i] <- pred_blist_for_year_var[[penalization_index]]
    } else {
      # If the index is out of bounds, assign NA
      best_r_prediction[j - ind_first + 1, i] <- NA
      print(paste("Warning: Penalization", best_penalization, "is out of bounds for the variable", rownames(MSFE_l_matrix)[i], "and year", j))
    }
  }
}

# Display the matrix of the best predictions for the selected years
print(best_r_prediction)

best_r_prediction <- as.data.frame(best_r_prediction)



path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_r_model, file = "Results/Best Models/best_r_model.rds")
saveRDS(best_r_prediction, file = "Results/Best Models/best_r_prediction.rds")




# ---- LASSO ----

# Initialize an empty list to save the results
best_l_model <- data.frame(Variable = rownames(MSFE_l_matrix),
                            Best_MSFE = numeric(nrow(MSFE_l_matrix)),
                            Best_Penalization = character(nrow(MSFE_l_matrix)),
                            nu_LASSO_l = numeric(nrow(MSFE_l_matrix)),
                            stringsAsFactors = FALSE)

# Create a mapping between the names of the penalties and the numerical indices
penalization_map <- c("1" = 1, "3" = 2, "5" = 3, "10" = 4, 
                      "25" = 5, "40" = 6, "50" = 7)

# Loop for every row of matrix MSFE
for (i in 1:nrow(MSFE_l_matrix)) {
  
  # Find the index of the column with the minimum value in row i
  min_index <- which.min(MSFE_l_matrix[i, ])
  
  # Assign the minimum value and the column name to the results list
  best_l_model$Best_MSFE[i] <- MSFE_l_matrix[i, min_index]
  best_l_model$Best_Penalization[i] <- colnames(MSFE_l_matrix)[min_index]
  
  #  Get the name of the best penalty for this variable
  best_penalization <- best_l_model$Best_Penalization[i]
  
  # Check if the penalty is present in the mapping
  if (best_penalization %in% names(penalization_map)) {
    
    # Get the index corresponding to the penalty
    penalization_index <- penalization_map[best_penalization]
    
    # Select the corresponding prediction from nu_LASSO
    best_l_model$nu_LASSO_l[i] <- nu_LASSO[[i]][[penalization_index]]
    
  } else {
    # Handle the case where the penalty is not found
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_l_matrix)[i]))
    best_l_model$nu_LASSO_l[i] <- NA  # # Assign NA if penalty not found
  }
}

#Visualize results
print(best_l_model)

# Initialize an empty matrix for the best predictions for each year.
best_l_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(LASSO))), ncol = nrow(MSFE_l_matrix))
rownames(best_l_prediction) <- seq(from = ind_first, to = length(LASSO))  
colnames(best_l_prediction) <- rownames(MSFE_l_matrix)  


# Loop for every variable (row of MSFE_l_matrix)
for (i in 1:nrow(MSFE_l_matrix)) {
  
  # Find the index of the column with the minimum value in row i (the optimal penalty)
  min_index <- which.min(MSFE_l_matrix[i, ])
  
  # Get the name of the best penalty for this variable
  best_penalization <- colnames(MSFE_l_matrix)[min_index]
  
  # Get the index of the corresponding penalty
  penalization_index <- penalization_map[best_penalization]
  
  # For each year (from ind_first to LASSO), select the prediction corresponding to the optimal penalty
  for (j in ind_first:length(LASSO)) {  # Itera sugli anni
    # Extract the list of predictions for variable i and year j from LASSO
    pred_blist_for_year_var <- LASSO[[j]][[i]]
    
    # Check if penalization_index is within the bounds of the predictions list
    if (penalization_index <= length(pred_blist_for_year_var)) {
      # Save the prediction corresponding to the best penalty parameter
      best_l_prediction[j - ind_first + 1, i] <- pred_blist_for_year_var[[penalization_index]]
    } else {
      #If the index is out of bounds, assign NA
      best_l_prediction[j - ind_first + 1, i] <- NA
      print(paste("Warning: Penalization", best_penalization, "is out of bounds for the variable", rownames(MSFE_l_matrix)[i], "and year", j))
    }
  }
}

# Display the matrix of the best predictions for the selected years
print(best_l_prediction)

best_l_prediction <- as.data.frame(best_l_prediction)


path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_l_model, file = "Results/Best Models/best_l_model.rds")
saveRDS(best_l_prediction, file = "Results/Best Models/best_l_prediction.rds")


# ---- PC ---- 

# Initialize an empty list to save the results
best_PC_model <- data.frame(Variable = rownames(MSFE_PC_matrix),
                            Best_MSFE = numeric(nrow(MSFE_PC_matrix)),
                            Best_Parameter = character(nrow(MSFE_PC_matrix)),
                            stringsAsFactors = FALSE)

# Create a mapping between the names of the penalties and the numerical indices
penalization_map <- c("1" = 1, "3" = 2, "5" = 3, "10" = 4, 
                      "25" = 5, "40" = 6, "50" = 7)

# Loop for every row of matrix MSFE
for (i in 1:nrow(MSFE_PC_matrix)) {
  
  # Find the index of the column with the minimum value in row i
  min_index <- which.min(MSFE_PC_matrix[i, ])
  
  # Assign the minimum value and the column name to the results list
  best_PC_model$Best_MSFE[i] <- MSFE_PC_matrix[i, min_index]
  best_PC_model$Best_Parameter[i] <- colnames(MSFE_PC_matrix)[min_index]
  
  # Get the name of the best penalty for this variable
  best_parameter <- best_PC_model$Best_Parameter[i]
  
  # Check if the penalty is present in the mapping
  if (best_parameter %in% names(penalization_map)) {
    
    # Get the index corresponding to the penalty
    penalization_index <- penalization_map[best_parameter]
    
  } else {
    # Handle the case where the penalty is not found
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_l_matrix)[i]))
    best_l_model$nu_LASSO_l[i] <- NA  # Assign NA if penalty not found
  }
}

# Visualize results
print(best_l_model)

# Initialize an empty matrix for the best predictions for each year
best_pc_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(PC))), ncol = nrow(MSFE_PC_matrix))
rownames(best_pc_prediction) <- seq(from = ind_first, to = length(PC))  
colnames(best_pc_prediction) <- rownames(MSFE_PC_matrix) 


# Loop for every variable (row of MSFE_PC_matrix)
for (i in 1:nrow(MSFE_PC_matrix)) {
  
  #Find the index of the column with the minimum value in row i (the optimal penalty)
  min_index <- which.min(MSFE_PC_matrix[i, ])
  
  # Get the name of the best penalty for this variable
  best_penalization <- colnames(MSFE_PC_matrix)[min_index]
  
  # Get the index of the corresponding penalty
  penalization_index <- penalization_map[best_penalization]
  
  # For each year (from ind_first to PC), select the prediction corresponding to the optimal penalty
  for (j in ind_first:length(PC)) {  # Itera sugli anni
    # Extract the list of predictions for variable i and year j from PC
    pred_blist_for_year_var <- PC[[j]][[i]]
    
    # Check if penalization_index is within the bounds of the predictions list
    if (penalization_index <= length(pred_blist_for_year_var)) {
      # Save the prediction corresponding to the best penalty parameter
      best_pc_prediction[j - ind_first + 1, i] <- pred_blist_for_year_var[[penalization_index]]
    } else {
      # If the index is out of bounds, assign NA
      best_pc_prediction[j - ind_first + 1, i] <- NA
      print(paste("Warning: Penalization", best_penalization, "is out of bounds for the variable", rownames(MSFE_l_matrix)[i], "and year", j))
    }
  }
}

# Display the matrix of the best predictions for the selected years
print(best_pc_prediction)

best_pc_prediction <- as.data.frame(best_pc_prediction)


path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_PC_model, file = "Results/Best Models/best_PC_model.rds")
saveRDS(best_pc_prediction, file = "Results/Best Models/best_pc_prediction.rds")


# ==============================================================================
# ================= VARIABLE FREQUENCY LASSO MODEL SELECTION  ===================
# ==============================================================================

# Initialize an empty matrix for the best predictions for each year
variable_selection <- matrix(NA, nrow = length(seq(from = start_sample, to = length(LASSO) - HH)), ncol = nrow(MSFE_l_matrix))
rownames(variable_selection) <- seq(from = start_sample, to = length(LASSO) - HH) 
colnames(variable_selection) <- rownames(MSFE_l_matrix) 

# Loop for every variable (row of MSFE_l_matrix)
for (i in 1:nrow(MSFE_l_matrix)) {
  
  # Find the index of the column with the minimum value in row i (the optimal penalty)
  min_index <- which.min(MSFE_l_matrix[i, ])
  
  # Get the name of the best penalty for this variable
  best_penalization <- colnames(MSFE_l_matrix)[min_index]
  
  # Get the index of the corresponding penalty
  penalization_index <- penalization_map[best_penalization]
  
  # For each year (from start_sample to LASSO), select the prediction corresponding to the optimal penalty
  for (j in start_sample:(length(LASSO) - HH)) {  
    # Extract the list of predictions for variable i and year j from LASSO
    pred_blist_for_year_var <- pred_bl[[j]][[i]]
    
    # Check that the list of predictions for the year and variable is valid
    if (is.list(pred_blist_for_year_var) && length(pred_blist_for_year_var) >= penalization_index) {
      # Select the model corresponding to the optimal penalty
      selected_model <- pred_blist_for_year_var[[penalization_index]]
      
      # Check if 'model_selection' is a list (it may contain multiple models)
      if ("model_selection" %in% names(selected_model)) {
        model_selection_values <- selected_model[["model_selection"]]
        
        # If 'model_selection' is a list or vector, you can decide how to select the values.
        if (length(model_selection_values) > 1) {
          # For example, take all the values from 'model_selection' and assign them 
          # You might also want to perform an aggregation, like the mean, if you want a single value
          variable_selection[j - start_sample + 1, i] <- paste(model_selection_values, collapse = ", ")
        } else {
          # If there is only one value, assign that value
          variable_selection[j - start_sample + 1, i] <- model_selection_values
        }
      } else {
        # If 'model_selection' is not present in the model, assign NA
        variable_selection[j - start_sample + 1, i] <- NA
        print(paste("Warning: 'model_selection' not found for the variable", rownames(MSFE_l_matrix)[i], "and year", j))
      }
    } else {
      # If the index is out of bounds or the list is not valid, assign NA
      variable_selection[j - start_sample + 1, i] <- NA
      print(paste("Warning: Penalization", best_penalization, "is out of bounds for the variable", rownames(MSFE_l_matrix)[i], "and year", j))
    }
  }
}


# Display the matrix of the best predictions for the selected years
print(variable_selection)

# Convert the matrix into long format
variable_selection_long <- as.data.frame(variable_selection) %>%
  mutate(Year = rownames(variable_selection)) %>%
  gather(key = "Variable", value = "SelectedModel", -Year)

# Count the frequency of selections for each target variable
freq_table <- variable_selection_long %>%
  group_by(Variable, SelectedModel) %>%
  summarise(Frequency = n(), .groups = 'drop')


#Convert to a DataFrame for manipulation
variable_selection_df <- as.data.frame(variable_selection)

# Make data "long" (one row for each selected variable)
freq_table <- variable_selection_df %>%
  pivot_longer(cols = everything(), names_to = "Target", values_to = "SelectedVariables") %>%
  # Split multiple variables into individual rows
  separate_rows(SelectedVariables, sep = ",") %>%
  # count the frequency for each selected variable
  count(Target, SelectedVariables, name = "Frequency") %>%
  # Sort the results by Target and frequency
  arrange(Target, desc(Frequency))


total_predictions <- ind_last - ind_first + 1
perc_freq_table <- freq_table %>%
  mutate(Percentage = Frequency / total_predictions * 100)

perc_freq_table[, 2] <- sapply(perc_freq_table[, 2], trimws, USE.NAMES = FALSE)

# Visualize table
print(perc_freq_table)

# Filter 5 most selected variables for every target 
top_5_per_target <- perc_freq_table %>%
  group_by(Target) %>%
  slice_max(Frequency, n = 5) %>%  
  ungroup()

top_5_per_target[, 2] <- sapply(top_5_per_target[, 2], trimws, USE.NAMES = FALSE)

# Transform into a list of matrices (one for each target variable)
top_5_matrices <- top_5_per_target %>%
  group_split(Target) %>%
  setNames(unique(top_5_per_target$Target)) %>%
  lapply(function(df) {
    matrix(data = c(df$SelectedVariables, df$Frequency, df$Percentage),
           ncol = 3,
           dimnames = list(NULL, c("Variable", "Frequency", "Percentage")))
  })

# Visualize matrices for every target
top_5_matrices$GDP  # Example: matrix for GDP
top_5_matrices$PPINRG # Example: matrix for Prices
top_5_matrices$WS  # Example: matrix for WS


LTX_top_5_per_target <- xtable(top_5_per_target)
print(LTX_top_5_per_target, type = "latex" )


# ==============================================================================
# ========================== EIGENGAP VISUALIZATION ============================ 
# ==============================================================================

# Initialize empty matrix for lagged predictors
temp <- NULL

# Create lagged predictors: X = [x, x_{-1}, ..., x_{-p}]
for (j in 0:p) {
  temp <- cbind(temp, x[(p + 1 - j):(nrow(x) - j), ])
}
X <- temp

# Adjust y to match the rows of X
# Uncomment if necessary
# y <- y[(p + 1):length(y)]

# Standardize predictors
X_standardized <- scale(X)

# Perform PCA on standardized predictors
pca_res <- prcomp(X_standardized, center = FALSE, scale. = FALSE)

# Calculate explained variance for each principal component
explained_variance <- pca_res$sdev^2

# Total variance for normalization
total_variance <- sum(explained_variance)

# Normalize variance and calculate cumulative variance
explained_variance_normalized <- explained_variance / total_variance * 100
cumulative_variance <- cumsum(explained_variance_normalized)

# Plot explained variance with ggplot
df_variance <- data.frame(PC = seq_along(explained_variance_normalized), Variance = explained_variance_normalized)
ggplot(df_variance, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.6, width = 0.7) +
  geom_line(aes(group = 1), color = "blue", size = 1) +
  geom_point(color = "blue", size = 2.5) +
  labs(
    title = "Explained Variance by Each Principal Component",
    x = "Principal Components (PC)",
    y = "Explained Variance (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

# Plot cumulative variance with ggplot
df_cumulative <- data.frame(PC = seq_along(cumulative_variance), CumulativeVariance = cumulative_variance)
ggplot(df_cumulative, aes(x = PC, y = CumulativeVariance)) +
  geom_line(color = "darkorange", size = 1) +
  geom_point(color = "darkorange", size = 3) +
  labs(
    title = "Cumulative Explained Variance by Principal Components",
    x = "Principal Components (PC)",
    y = "Cumulative Explained Variance (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )


# Interpretations of PC 

pca_res$rotation[,c(1,2,3)]




