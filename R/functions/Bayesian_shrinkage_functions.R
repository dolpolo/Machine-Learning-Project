getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project/R/functions"
setwd(path)

# **************************************************************************** #
# *********************** BAYEASIAN SHRINKAGE FUNCTIONS ********************** #
# **************************************************************************** #  
  
  

# ============================================================================ #
# =============================== RIDGE FUNCTIONS ============================ # 
# ============================================================================ #

# This section will provide the function to evaluate the bayasian shrinkage mathod under 
# Gaussian prior. Under this condition the posterior mode is maximized using the ridge

  
################################## RIDGE PREDICT ###############################

# This function will provide the parameters under a Gaussian Prior

RIDGE_pred <- function(y, x, p, nu, h) {
  
  # Initialize an empty matrix to store lagged predictors
  temp <- NULL
  
  # Create lagged predictors: X = [x, x_{-1}, ..., x_{-p}]
  for (j in 0:p) {
    temp <- cbind(temp, x[(p + 1 - j):(nrow(x) - j), ])
  }
  X <- temp
  
  # Trim the dependent variable to match the lagged predictors
  y <- y[(p + 1):length(y)]
  
  # Normalize the predictors to have |X| < 1
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  # Prepare the predictors for regression: Z = [XX(1:end-h, :)]
  Z <- XX[1:(nrow(XX) - h), ]
  
  # Compute the dependent variable for the forecast: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Standardize the dependent variable
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # Ridge regression: b = inv(Z'Z + nu * I) * Z'y_std
  Z_transpose <- t(Z)
  ridge_coeff <- solve(Z_transpose %*% Z + nu * diag(N)) %*% (Z_transpose %*% y_std)
  
  # Forecast: pred = XX(end, :) * b * sy + my
  pred <- (XX[nrow(XX), ] %*% ridge_coeff) * sy + my
  
  # Calculate the in-sample variance explained by the regression (MSE)
  MSE <- var(y_std - Z %*% ridge_coeff, na.rm = TRUE)
  
  return(list(pred = pred, b = ridge_coeff, MSE = MSE))
}


# Example usage:
# y <- dependent variable (vector)
# x <- predictors matrix
# p <- number of lags
# nu <- penalization parameter for Ridge regression
# h <- number of steps ahead to predict

# result <- RIDGE_pred(y, x, p, nu, h)
# print(result$pred)  # Predicted value
# print(result$b)     # Ridge regression coefficients
# print(result$MSE)   # In-sample variance explained (MSE)




################################# RIDGE SET ####################################

# This function will select the optimal penalization parameter for each in sample variance

SET_ridge <- function(y, x, p, INfit, h) {
  
  nu_min <- 0
  nu_max <- 10 #higher degree of penalization
  IN_max <- 1e+32 
  IN_min <- 0
  IN_avg <- 1e+32
  
  while (abs(IN_avg - INfit) > 1e-7) {
    
    nu_avg <- (nu_min + nu_max) / 2
    
    result <- RIDGE_pred(y, x, p, nu_avg, h)
    pred <- result$pred
    b <- result$b
    IN_avg <- result$MSE
    
    if (IN_avg > INfit) {
      nu_min <- nu_min
      nu_max <- nu_avg
    } else {
      nu_min <- nu_avg
      nu_max <- nu_max
    }
  }
  
  nu <- nu_avg
  b <- b[, ncol(b)] # variable aggregation
  
  return(list(nu = nu, b = b))
}




# ==============================================================================
# =============================== LASSO FUNCTIONS ============================== 
# ==============================================================================

# This section will provide the function to evaluate the bayasian shrinkage mathod under 
# double esponential prior. Under this condition the posterior mode is maximized using the LASSO


################################## LASSO PREDICT ###############################

# This function will provide the parameters under a double exponential prior

LASSO_pred <- function(y, x, p, nu, h) {
  # Initialize an empty matrix to store lagged predictors
  temp <- NULL
  
  # Create lagged predictors: X = [x, x_{-1}, ..., x_{-p}]
  for (j in 0:p) {
    temp <- cbind(temp, x[(p + 1 - j):(nrow(x) - j), ])
  }
  X <- temp
  
  # Trim the dependent variable to match the lagged predictors
  y <- x[, nn[1]][(p + 1):length(x[, nn[1]])]
  
  # Normalize the predictors to have |X| < 1
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  h=4
  # Prepare the predictors for regression: Z = [XX(1:end-h, :)]
  Z <- X[1:(nrow(XX) - h), ]
  
  # Compute the dependent variable for the forecast: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Standardize the dependent variable
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  lasso_model <- glmnet(Z, y_std, alpha = 1, standardize=FALSE, lambda=nu)
  pred <- predict(lasso_model, newx = matrix(X[nrow(X), ], nrow = 1)) * sy + my
  
  # consistency in variable selection
  # Identify the selected variables (non-zero coefficients)
  lasso_coeff <- as.matrix(lasso_model$beta[,1])
  selected_variables <- rownames(lasso_coeff)[lasso_coeff != 0]
  
  return(list(pred = pred, model_selection = selected_variables))
  
}

##################################  SET LASSO ##################################

SET_lasso <- function(y, x, p, K, h) {
  
  # Initialize an empty matrix to store lagged predictors
  temp <- NULL
  
  # Create lagged predictors: X = [x, x_{-1}, ..., x_{-p}]
  for (j in 0:p) {
    temp <- cbind(temp, x[(p + 1 - j):(nrow(x) - j), ])
  }
  X <- temp
  
  # Trim the dependent variable to match the lagged predictors
  y <- y[(p + 1):length(y)]
  
  # Normalize the predictors to have |X| < 1
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  # Prepare the predictors for regression: Z = [XX(1:end-h, :)]
  Z <- X[1:(nrow(X) - h), ]
  
  # Compute the dependent variable for the forecast: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Standardize the dependent variable
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  lasso_model <- glmnet(Z, y_std, alpha = 1, standardize=FALSE, nlambda = 1500, lambda.min.ratio=0.00001)
  lasso_df <- as.data.frame(cbind(lasso_model[["df"]],lasso_model[["lambda"]]))
  if (K <= 10) {
    nu <- lasso_df %>%
      filter(V1 == K) %>%
      filter(V2 == min(V2)) %>% 
      pull(V2)  # Extract only the value of V2
  } else {
    nu <- lasso_df %>%
      filter(V1 == K) %>%
      filter(V2 == max(V2)) %>%
      pull(V2)  # Extract only the value of V2
  }
  return(list(nu = nu))
}



# ==============================================================================
# ====================== PRINCIPAL COMPONENT FUNCTIONS =========================
# ==============================================================================


# This function will create lagged predictors, computing principal components for an h-step-ahead forecasts.

PC_pred <- function(y, x, p, r, h) {
  
  # Initialize empty list to store lagged predictors
  temp <- NULL
  
  # Create lagged predictors: X = [x, x_{-1}, ..., x_{-p}]
  for (j in 0:p) {
    temp <- cbind(temp, x[(p + 1 - j):(nrow(x) - j), ])
  }
  X <- temp
  
  # Adjust y to match the rows of X
  y <- y[(p + 1):length(y)]
  
  # Standardize predictors
  XX <- scale(X)
  
  # Compute the principal components on standardized predictors
  pca_res <- prcomp(XX, center = FALSE, scale. = FALSE)  # Use standardized data
  F <- pca_res$x[, r]  # Principal components
  Z <- cbind(1, F)  # Add intercept
  
  # Compute the dependent variable to predict as the h-step ahead average
  Y <- stats::filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]  # Adjust length of Y
  
  # Fit regression model for forecasting
  gamma <- solve(t(Z[1:(nrow(Z) - h), ]) %*% Z[1:(nrow(Z) - h), ]) %*% t(Z[1:(nrow(Z) - h), ]) %*% Y
  
  # Compute predictions
  pred <- Z[(nrow(Z)), ] %*% gamma
  
  return(list(pred = pred))
}


# ==============================================================================
# =============================== GENERAL FUNCTIONS ============================
# ==============================================================================

# Function to initialize a nested list structure
initialize_nested_list <- function(outer_length, inner_lengths) {
  # Recursive function to initialize a list with specified lengths at each level
  if (length(inner_lengths) == 0) {
    return(vector("list", length = outer_length))
  } else {
    return(lapply(1:outer_length, function(x) initialize_nested_list(inner_lengths[1], tail(inner_lengths, -1))))
  }
}


# Function to initialize lists for each dependent variable and time interval
initialize_output_list <- function(variable_count, interval_length) {
  output_list <- vector("list", length = variable_count)
  for (i in 1:variable_count) {
    output_list[[i]] <- vector("list", length = interval_length)
  }
  return(output_list)
}

# Function to process and store matrices for each variable and interval
process_output <- function(data, output_list, ind_first, variable_count) {
  for (j in ind_first:length(data)) {
    for (k in 1:variable_count) {
      # Retrieve predictions or true values for the current year and variable
      quarter_data <- data[[j]][[k]]
      # Combine the data into a matrix
      quarter_matrix <- do.call(rbind, lapply(quarter_data, unlist))
      # Store the matrix in the output list
      output_list[[k]][[j - ind_first + 1]] <- quarter_matrix
    }
  }
  return(output_list)
}

# Function to calculate squared differences between true and predicted values
calculate_squared_differences <- function(true_output, predicted_output, variable_count, interval_length) {
  diff_output <- initialize_output_list(variable_count, interval_length)
  
  for (k in 1:variable_count) {
    for (j in 1:interval_length) {
      true_vector <- true_output[[k]][[j]]
      predicted_vector <- predicted_output[[k]][[j]]
      
      # Ensure vectors are the same length before calculating differences
      if (length(true_vector) == length(predicted_vector)) {
        diff_output[[k]][[j]] <- (true_vector - predicted_vector)^2
      }
    }
  }
  return(diff_output)
}

# Function to calculate the mean squared forecast error (MSFE) for each variable
calculate_msfe <- function(diff_output, variable_count) {
  msfe_output <- vector("list", length = variable_count)
  
  for (k in 1:variable_count) {
    vectors_k <- diff_output[[k]]
    vector_length <- length(vectors_k[[1]]) # Assuming all vectors have the same length
    mean_vector <- numeric(vector_length)
    
    for (pos in 1:vector_length) {
      # Extract elements at position 'pos' across all years
      elements_at_pos <- sapply(vectors_k, function(v) v[pos])
      mean_vector[pos] <- mean(elements_at_pos, na.rm = TRUE)
    }
    
    msfe_output[[k]] <- mean_vector
  }
  return(msfe_output)
}

# Function to convert a list of vectors to a named data frame matrix
convert_to_named_matrix_r <- function(msfe_list, target_var, col_prefix) {
  vector_lengths <- sapply(msfe_list, length)
  
  # Check if all vectors have the same length
  if (length(unique(vector_lengths)) == 1) {
    msfe_matrix <- do.call(rbind, msfe_list)
    msfe_matrix <- as.data.frame(msfe_matrix)
    
    # Set row and column names
    rownames(msfe_matrix) <- target_var
    colnames(msfe_matrix) <- INfit
    
    return(msfe_matrix)
  } else {
    stop("All vectors in the list must have the same length.")
  }
}

# Function to convert a list of vectors to a named data frame matrix
convert_to_named_matrix <- function(msfe_list, target_var, col_prefix) {
  vector_lengths <- sapply(msfe_list, length)
  
  # Check if all vectors have the same length
  if (length(unique(vector_lengths)) == 1) {
    msfe_matrix <- do.call(rbind, msfe_list)
    msfe_matrix <- as.data.frame(msfe_matrix)
    
    # Set row and column names
    rownames(msfe_matrix) <- target_var
    colnames(msfe_matrix) <- K
    
    return(msfe_matrix)
  } else {
    stop("All vectors in the list must have the same length.")
  }
}


# ==============================================================================
# =========================== RANDOM WALK FUNCTIONS ============================
# ==============================================================================


RW_pred <- function(y, h) {
  # Compute h-step ahead prediction from constant growth model
  # Input:
  # y: dependent variable
  # h: number of steps ahead
  
  # Output:
  # pred: h-step ahead predictions
  
  # Compute the variable to be predicted
  Y <- stats::filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Compute the prediction
  pred <- mean(Y, na.rm = TRUE)
  
  return(pred)
}
