# ---- Set Directory
getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)

# ---- Libraries 
library(dplyr)
library(tidyverse)
library(tseries)
library(ggplot2)
library(writexl)
library(readxl)
library(lubridate)
library(glmnet)


# **************************************************************************** #
# ********************************** FarmSelect ****************************** #
# **************************************************************************** #  


# ---- LOAD DATA ----
# load EA_data untill 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

# ---- CALL FUNCTIOINS ----

# lapply(list.files("R/functions/", full.names = TRUE), source)
source("R/functions/FarmSelect_functions.R")


# ---- SET PARAMETERS ----
# Dependent variables to be predicted
nn <- c(1,37,97)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 25, 40, 50)  # Number of principal components
K <- rr  # Number of predictors with LASSO
HH <- c(4)  # Steap-ahead prediction
Jwind <- 56  # Rolling window


# ********************************* IN SAMPLE ******************************** #

# Initial date out-of-sample
start_y <- 2014
start_m <- 01

DATA <- as.matrix(EAdataQ[, -1])  
series <- colnames(EAdataQ)
X <- DATA

# target variables
target_var <- colnames(X)[nn]


TT <- nrow(X)
NN <- ncol(X)


start_sample <- which(year(EAdataQ$Time) == start_y & month(EAdataQ$Time) == start_m)
if (Jwind > start_sample) stop("The rolling window cannot be larger than the first evaluation sample")

j0 <- start_sample - Jwind + 1
x <- X[j0:start_sample, ]



# ==============================================================================
# ============================= LASSO PENALIZATION ============================= 
# ==============================================================================

nu_lasso <- list()

for (jK in seq_along(K)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      
      if (is.null(nu_lasso[[paste(h, k, sep = "_")]])) {
        nu_lasso[[paste(h, k, sep = "_")]] <- numeric(length(K))  
      }
      
      # Compute `nu` value using `SET_FarmSelect`
      nu_result <- SET_FarmSelect(
        y = x[, nn[k]],
        x = x,
        p = p,
        K = K[jK],
        h = h
      )$nu
      
      # Assign the result to `nu_lasso`
      nu_lasso[[paste(h, k, sep = "_")]][jK] <- if (!is.null(nu_result) && length(nu_result) > 0) nu_result else NA
    }
  }
}
nu_lasso
# ================================= out of sample ==============================

# ==============================================================================
# =============================== LASSO PREDICTION ============================= 
# ==============================================================================

# Define lengths for each level
outer_length <- TT
mid_length <- length(nn)
inner_length <- length(K)

# Initialize each list using the function
pred_FS <- initialize_nested_list(outer_length, c(mid_length, inner_length))
FarmSelect <- initialize_nested_list(outer_length, c(mid_length, inner_length))
true_FS <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RW_FS <- initialize_nested_list(outer_length, c(mid_length, inner_length))

# Out of sample LASSO
for (j in start_sample:(TT - HH)) {
  
  
  j0 <- j - Jwind + 1 # Starting point for rolling window 
  
  # Data available at every evaluation point
  x <- X[j0:j, ]  
  
  for (k in seq_along(nn)) {
    
    for (jK in seq_along(K)) {
      
     
      const <- 4
      
      # Farmselect prediction
      for (h in HH) {  
        
        # Prediction FarmSelect with FarmSelect_pred
        pred_FS[[j]][[k]][[jK]] <- FarmSelect_pred(
          y = x[, nn[k]], 
          x = x, 
          p = p, 
          nu = nu_lasso[[paste(h, k, sep = "_")]][jK], # optimal nu
          h = h
        )
        
        # Save prediction with normalization
        FarmSelect[[j+h]][[k]][[jK]] <- pred_FS[[j]][[k]][[jK]][["pred"]] * const
        
        # Compute true value to forecast 
        temp <- mean(X[(j + 1):(j + h), nn[k]], na.rm = TRUE)
        true_FS[[j+h]][[k]][[jK]] <- temp * const
        
        # Prediction RW with constant growth 
        temp <- RW_pred(x[, nn[k]], h)
        RW_FS[[j+h]][[k]][[jK]] <- temp * const
      }
    }
  }
}



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


# ************************ FarmSelect MSFE AND VISUALIZATION *******************

# Turn the prediction for each variable into a matrix

# ---- MSFE computation

# Number of variables and length of intervals
variable_count <- length(nn)
interval_length <- length(ind_first:length(FarmSelect))

# Initialize output lists
FarmSelect_output <- initialize_output_list(variable_count, interval_length)
true_FS_output <- initialize_output_list(variable_count, interval_length)
RW_output <- initialize_output_list(variable_count, interval_length)

# Process FarmSelect predictions
FarmSelect_output <- process_output(FarmSelect, FarmSelect_output, ind_first, variable_count)

# Process true values
true_FS_output <- process_output(true_FS, true_FS_output, ind_first, variable_count)

# Process Random Walk predictions
RW_output <- process_output(RW_FS, RW_output, ind_first, variable_count)

# Calculate squared differences
diff_FS_output <- calculate_squared_differences(true_FS_output, FarmSelect_output,
                                                variable_count, interval_length)
diff_RW_output <- calculate_squared_differences(true_FS_output, RW_output, variable_count,
                                                interval_length)

# Calculate mean squared forecast error (MSFE)
MSFE_FS <- calculate_msfe(diff_FS_output, variable_count)
MSFE_RW <- calculate_msfe(diff_RW_output, variable_count)

# Display results
print(MSFE_FS)
print(MSFE_RW)

# ---- MFSE Ratio

# Convert MSFE_FS and MSFE_RW lists to matrices
MSFE_FS_matrix <- convert_to_named_matrix(MSFE_FS, target_var, K)
print(MSFE_FS_matrix)

MSFE_RW_matrix <- convert_to_named_matrix(MSFE_RW, target_var, K)
print(MSFE_RW_matrix)

# Calculate and display the ratio of MSFE_FS to MSFE_RW
MSFE_FS_ratio <- MSFE_FS_matrix / MSFE_RW_matrix
print(MSFE_FS_ratio)


# ---- Visualization

# Convert dataframe into long format, includying identifier for rows
MSFE_FS_matrix_long <- MSFE_FS_matrix %>%
  rownames_to_column(var = "Variable") %>%
  pivot_longer(cols = -Variable, names_to = "Penalization", values_to = "Value")

MSFE_FS_matrix_long <- MSFE_FS_matrix_long %>%
  mutate(Penalization = factor(Penalization, levels = unique(Penalization)))

ggplot(MSFE_FS_matrix_long, aes(x = Penalization, y = Value, color = Variable, group = Variable)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  labs(
    title = "MSFE LASSO FarmSelect",
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


# ==============================================================================
# =================================== BEST MODEL ===============================
# ==============================================================================

# Initialize empty dataframe to save results
best_FS_model <- data.frame(Variable = rownames(MSFE_FS_matrix),
                            Best_MSFE = numeric(nrow(MSFE_FS_matrix)),
                            Best_Penalization = character(nrow(MSFE_FS_matrix)),
                            nu_lasso_FS = numeric(nrow(MSFE_FS_matrix)),
                            stringsAsFactors = FALSE)

# Create a mapping between the names of the penalties and the numerical indices
penalization_map <- c("1" = 1, "3" = 2, "5" = 3, "10" = 4, 
                      "15" = 5, "40" = 6, "50" = 7)

# Loop for every row of matrix MSFE
for (i in 1:nrow(MSFE_FS_matrix)) {
  
  min_index <- which.min(MSFE_FS_matrix[i, ])
  best_FS_model$Best_MSFE[i] <- MSFE_FS_matrix[i, min_index]
  best_FS_model$Best_Penalization[i] <- colnames(MSFE_FS_matrix)[min_index]
  best_penalization <- best_FS_model$Best_Penalization[i]

  if (best_penalization %in% names(penalization_map)) {
 
    penalization_index <- penalization_map[best_penalization]
    
    # Select corresponding prediction through nu lasso
    best_FS_model$nu_lasso_FS[i] <- nu_lasso[[i]][[penalization_index]]
    
  } else {
    # Case where penalization not found
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_FS_matrix)[i]))
    best_FS_model$nu_lasso_FS[i] <- NA  #NA if nu not found 
  }
}

# Visualize results 
print(best_FS_model)

# Initialize empty matrix for best predicitions for every year 
best_FS_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(FarmSelect))), ncol = nrow(MSFE_FS_matrix))
rownames(best_FS_prediction) <- seq(from = ind_first, to = length(FarmSelect))  
colnames(best_FS_prediction) <- rownames(MSFE_FS_matrix)  


# Loop for every variable (row MSFE_FS_matrix)
for (i in 1:nrow(MSFE_FS_matrix)) {

  min_index <- which.min(MSFE_FS_matrix[i, ])
  best_penalization <- colnames(MSFE_FS_matrix)[min_index]
  penalization_index <- penalization_map[best_penalization]
  
  # For every year (from ind_first to FarmSelect), select prediction corresponding to optimal penalization 
  for (j in ind_first:length(FarmSelect)) { 
    # Extract list of predictions for variable i and year j from FarmSelect
    pred_list_for_year_var <- FarmSelect[[j]][[i]]
    
    # Check if penalization_index is within limits of list of prediction
    if (penalization_index <= length(pred_list_for_year_var)) {
      # Save prediction corresponding to best nu
      best_FS_prediction[j - ind_first + 1, i] <- pred_list_for_year_var[[penalization_index]]
    } else {
      # If index is out of bounds, assign NA
      best_FS_prediction[j - ind_first + 1, i] <- NA
      print(paste("Warning: Penalization", best_penalization, "It is out of bounds for the variable", rownames(MSFE_FS_matrix)[i], "e l'anno", j))
    }
  }
}

# Visualize the matrix of best predictions for selected years
print(best_FS_prediction)

best_FS_prediction <- as.data.frame(best_FS_prediction)



path <- getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"

setwd(path)
saveRDS(best_FS_model, file = "Results/Best Models/best_FS_model.rds")
saveRDS(best_FS_prediction, file = "Results/Best Models/best_FS_prediction.rds")



# ==============================================================================
# ====== VARIABLE FREQUENCY LASSO MODEL SELECTION ============
# ==============================================================================

# Initialize empty matrix for best predictions for every year 
variable_selction <- matrix(NA, nrow = length(seq(from = start_sample, to = length(FarmSelect) - HH)), ncol = nrow(MSFE_FS_matrix))
rownames(variable_selction) <- seq(from = start_sample, to = length(FarmSelect) - HH)  
colnames(variable_selction) <- rownames(MSFE_FS_matrix)  

# Loop for every variable (row of MSFE_FS_matrix)
for (i in 1:nrow(MSFE_FS_matrix)) {
  
  min_index <- which.min(MSFE_FS_matrix[i, ])
  best_penalization <- colnames(MSFE_FS_matrix)[min_index]
  penalization_index <- penalization_map[best_penalization]
  
  for (j in start_sample:(length(FarmSelect) - HH)) {  
    
    pred_list_for_year_var <- pred_FS[[j]][[i]]
    if (is.list(pred_list_for_year_var) && length(pred_list_for_year_var) >= penalization_index) {
      # Select model corrisponding to optimal nu
      selected_model <- pred_list_for_year_var[[penalization_index]]
      
      # Check if 'model_selection' is a list 
      if ("model_selection" %in% names(selected_model)) {
        model_selection_values <- selected_model[["model_selection"]]
        
        # If 'model_selection' is a list or a vector, you can decide how to select values 
        if (length(model_selection_values) > 1) {
          # For example, take all values of 'model_selection' and assign them
          # You could also perform an aggregation, like the mean, if you want a single value 
          variable_selction[j - start_sample + 1, i] <- paste(model_selection_values, collapse = ", ")
        } else {
          # If there is only one value, assign that value
          variable_selction[j - start_sample + 1, i] <- model_selection_values
        }
      } else {
        # if 'model_selection' not present in the model, assign NA
        variable_selction[j - start_sample + 1, i] <- NA
        print(paste("Warning: 'model_selection' not found for the variable", rownames(MSFE_FS_matrix)[i], "e l'anno", j))
      }
    } else {
      # If the index is out of range or the list is invalid, assign NA
      variable_selction[j - start_sample + 1, i] <- NA
      print(paste("Warning: Penalization", best_penalization, "is out of range for the variable", rownames(MSFE_FS_matrix)[i], "e l'anno", j))
    }
  }
}


# Visualize the matrix of best predictions for the years selected
print(variable_selction)

# Transform into dataframe 
variable_selection_df <- as.data.frame(variable_selction)

# Make data "long" (one row for every variable selected)
freq_table <- variable_selection_df %>%
  pivot_longer(cols = everything(), names_to = "Target", values_to = "SelectedVariables") %>%
  # Split multiple variables into single lines
  separate_rows(SelectedVariables, sep = ",") %>%
  # Compute frequency for selected variable
  count(Target, SelectedVariables, name = "Frequency") %>%
  # Order results for Target and frequency
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

LTX_top_5_per_target <- xtable(top_5_per_target)
print(LTX_top_5_per_target, type = "latex" )