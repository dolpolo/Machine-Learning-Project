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
  y <- y[(p + 1):length(y)]
  
  
  # Prepare the predictors for regression: Z = [XX(1:end-h, :)]
  Z <- X[1:(nrow(X) - h), ]
  
  # Compute the dependent variable for the forecast: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Standardize the dependent variable
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  lasso_model <- glmnet(Z, y_std, alpha = 1)
  lasso_coeff<- coef(lasso_model, s = nu)
  
  pred <- predict(lasso_model, newx = matrix(X[nrow(X), ], nrow = 1), s = nu) * sy + my
  
  
  return(list(pred = pred, b = lasso_coeff))
  
}

##################################  SET LASSO ##################################

SET_lasso <- function(y, x, p, K, h) {
  # Initialize parameters
  nu_min <- 0
  nu_max <- 2
  K_max <- 1e+32
  K_min <- 0
  K_avg <- 1e+32
  max_iter <- 30
  cont <- 0
  
  while (K_min != K_max && cont < max_iter) {
    cont <- cont + 1
    nu_avg <- (nu_min + nu_max) / 2
    
    # Call the LASSO prediction function
    result <- LASSO_pred(y, x, p, nu_avg, h)
    pred <- result$pred
    b <- result$b
    
    K_avg <- sum(b[, ncol(b)] != 0)  # Count non-zero coefficients
    
    if (K_avg < K) {
      nu_min <- nu_min
      nu_max <- nu_avg
    } else {
      nu_min <- nu_avg
      nu_max <- nu_max
    }
  }
  
  if (cont >= max_iter) {
    warning("Max iterations reached when setting the Lasso penalization")
  }
  
  nu <- nu_avg
  b <- b[, ncol(b)]  # Get the last column of b
  return(list(nu = nu, b = b))
}

# ==============================================================================
# ================================ LARS FUNCTIONS ============================== 
# ==============================================================================

LARS_pred <- function(y, x, p, h) {
  # Creazione di un vettore temporaneo per i predittori e i loro lag
  temp <- NULL
  
  # Mettere insieme i predittori e i loro lag: X = [x x_{-1} ... x_{-p}]
  for (j in 0:p) {
    temp <- cbind(temp, x[(p + 1 - j):nrow(x), , drop = FALSE])
  }
  
  # Definire X e normalizzare y
  X <- temp
  y <- y[(p + 1):length(y)]
  
  # Normalizzare i regressori per avere |X| < 1
  T <- nrow(X)
  N <- ncol(X)
  
  # Centering e scaling
  X_centered <- scale(X, center = TRUE, scale = FALSE)
  X_normalized <- X_centered / sqrt(N * T)
  
  # Regressori usati per il calcolo dei coefficienti di regressione
  Z <- X_normalized[1:(T - h), , drop = FALSE]
  
  # Calcolare la variabile dipendente da predire: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- filter(y, rep(1 / h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Standardizzare la variabile dipendente
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # Calcolo dei coefficienti di regressione usando lars
  beta <- lars(Z, y_std)
  
  # Calcolare le previsioni
  pred <- (tail(X_normalized, 1) %*% beta$beta) * sy + my
  
  return(list(pred = pred, beta = beta$beta))
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



#################################

PCA_pred <- function(y, x, p, r, h) {
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
  
  # Fit the PCR model
  PCR_model <- plsr(y_std ~ Z, ncomp = r, center = FALSE)
  
  # Extract coefficients
  pcr_coeff <- coef(PCR_model, ncomp = r)
  
  # Predict the last observation
  pred <- predict(PCR_model, newdata = matrix(X[nrow(X), ], nrow = 1), ncomp = r) * sy + my
  
  return(list(pred = pred, b = pcr_coeff))
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
