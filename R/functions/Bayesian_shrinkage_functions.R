getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project/R/functions"
setwd(path)

# **************************************************************************** #
# *********************** BAYEASIAN SHRINKAGE FUNCTIONS ********************** #
# **************************************************************************** #  
  
  


#================================ RIDGE FUNCTIONS ============================== 

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





#================================ LASSO FUNCTIONS ============================== 

# This section will provide the function to evaluate the bayasian shrinkage mathod under 
# double esponential prior. Under this condition the posterior mode is maximized using the LASSO


################################## LASSO PREDICT ###############################

# This function will provide the parameters under a double exponential prior

LASSO_pred <- function(y, x, p, nu, h, b_init = NULL) {
  
  # Creazione di una matrice temporanea vuota
  temp <- NULL
  
  # Creazione dei predittori con i loro lag
  for (j in 0:p) {
    temp <- cbind(temp, x[(p + 1 - j):(nrow(x) - j), ])
  }
  X <- temp
  y <- y[(p + 1):length(y)]
  
  # Normalizzazione dei regressori |X| < 1
  N <- ncol(X)
  T <- nrow(X)
  XX <- scale(X, center = TRUE, scale = TRUE) / sqrt(N * T)
  
  # Regressori usati per calcolare i coefficienti di regressione
  Z <- XX[1:(nrow(XX) - h), ]
  
  # Calcolo della variabile dipendente da predire
  Y <- filter(y, rep(1/h, h), sides = 1)
  Y <- Y[(h + 1):length(Y)]
  
  # Standardizzazione della variabile dipendente
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # Inizializzazione dell'algoritmo Iterative Landweber
  thresh <- nu / 2
  tolerance <- 1e-5
  max_iter <- 10000
  cont <- 1
  pred <- 0
  fit_prev <- 1e+32
  Dfit <- 1e+32
  
  # Inizializzazione dei parametri
  if (!is.null(b_init)) {
    b <- b_init
  } else {
    b <- matrix(0, N, 1)
  }
  
  # Iterative Landweber con soft thresholding
  while (Dfit > tolerance && cont < max_iter) {
    cont <- cont + 1
    b_temp <- matrix(0, N, 1)
    
    # Landweber iteration
    temp <- (b - t(Z) %*% Z %*% b + t(Z) %*% y_std)
    
    # Soft thresholding
    keep <- abs(temp) > thresh
    b_temp[keep] <- temp[keep] - thresh * sign(temp[keep])
    
    # Calcolo delle previsioni
    pred <- (XX[nrow(XX), ] %*% b_temp) * sy + my
    
    # Calcolo dell'errore quadratico medio in-sample
    fit <- var(y_std - Z %*% b_temp)
    
    # Verifica della convergenza
    Dfit <- abs(fit - fit_prev)
    fit_prev <- fit
    
    b <- cbind(b, b_temp)
  }
  
  if (cont == max_iter) {
    message("LASSO: Maximum iteration reached")
  }
  
  return(list(pred = pred, b = b))
}

##################################  SET LASSO ##################################

# This function will select the optimal penalization parameter for each in sample variance

SET_lasso <- function(y, x, p, K, h) {
  
  # Inizializzazione dei parametri
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
    
    # Chiamata alla funzione LASSO_pred
    lasso_result <- LASSO_pred(y, x, p, nu_avg, h)
    pred <- lasso_result$pred
    b <- lasso_result$b
    
    # Conta il numero di coefficienti non nulli
    K_avg <- sum(b[, ncol(b)] != 0)
    
    # Aggiornamento dei limiti su nu
    if (K_avg < K) {
      nu_max <- nu_avg
    } else {
      nu_min <- nu_avg
    }
  }
  
  if (cont >= max_iter) {
    warning("Warning: max iterations reached when setting the Lasso penalization")
  }
  
  nu <- nu_avg
  b <- b[, ncol(b)]
  
  return(list(nu = nu, b = b))
}
