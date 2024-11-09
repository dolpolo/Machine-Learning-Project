getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project/R/functions"
setwd(path)

# **************************************************************************** #
# **************************** FarmSelect FUNCTIONS ************************** #
# **************************************************************************** # 

# This section will provide the function to perform model selection through a 
# Factor adjusted regularized method. 


###########################
## FarmSelect prediction ##
###########################


FarmSelect_pred <- function(y, x, p, nu, h) {
  # Initialize lagged predictors matrix
  X <- do.call(cbind, lapply(0:p, function(j) x[(p + 1 - j):(nrow(x) - j), ]))
  
  # Standardize the predictors matrix
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  # Adjust the dependent variable y to match the lagged predictors
  y <- y[(p + 1):length(y)]
  
  # Compute forecast-dependent variable: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)[(h + 1):length(y)]
  
  # Standardize the dependent variable
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # PCA analysis on the standardized predictors
  pca_res <- prcomp(XX, center = FALSE)
  
  # Extract first principal component scores and loadings
  first_pc_scores <- pca_res$x[, 1]
  first_pc_loadings <- pca_res$rotation[, 1]
  
  # Calculate high correlation component
  first_pc_matrix <- matrix(first_pc_scores, nrow = length(first_pc_scores), ncol = 1)
  high_corr <- first_pc_matrix %*% t(first_pc_loadings)
  
  # Remove high correlation component to get low correlation predictors
  low_corr <- XX - high_corr
  
  # Prepare the predictors for regression
  Z <- low_corr[1:(nrow(low_corr) - h), ]
  
  # Fit the Lasso model
  FarmSelect_model <- glmnet(Z, y_std, alpha = 1, standardize = FALSE, lambda = nu)
  
  # Make predictions and revert to the original scale
  pred <- predict(FarmSelect_model, newx = matrix(low_corr[nrow(low_corr), ], nrow = 1)) * sy + my
  
  # Identify selected variables (non-zero coefficients)
  selected_variables <- rownames(FarmSelect_model$beta)[FarmSelect_model$beta[, 1] != 0]
  
  return(list(pred = pred, model_selection = selected_variables))
}


####################
## SET_FarmSelect ##
####################

SET_FarmSelect <- function(y, x, p, K, h) {
  # Initialize lagged predictors matrix
  X <- do.call(cbind, lapply(0:p, function(j) x[(p + 1 - j):(nrow(x) - j), ]))
  
  # Standardize the predictors matrix
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  # Adjust the dependent variable y to match the lagged predictors
  y <- y[(p + 1):length(y)]
  
  # Prepare the predictors for regression
  Z <- X[1:(nrow(X) - h), ]
  
  # Compute forecast-dependent variable: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)[(h + 1):length(y)]
  
  # Standardize the dependent variable
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # PCA analysis on the standardized predictors
  pca_res <- prcomp(XX, center = FALSE)
  
  # Extract first principal component scores and loadings
  first_pc_scores <- pca_res$x[, 1]
  first_pc_loadings <- pca_res$rotation[, 1]
  
  # Calculate high correlation component
  first_pc_matrix <- matrix(first_pc_scores, nrow = length(first_pc_scores), ncol = 1)
  high_corr <- first_pc_matrix %*% t(first_pc_loadings)
  
  # Remove high correlation component to get low correlation predictors
  low_corr <- XX - high_corr
  Z <- low_corr[1:(nrow(low_corr) - h), ]
  
  # Fit the Lasso model
  FarmSelect_model <- glmnet(Z, y_std, alpha = 1, standardize = FALSE, nlambda = 500, lambda.min.ratio = 0.00001)
  
  # Extract the degrees of freedom and lambda values
  FARM_df <- as.data.frame(cbind(FarmSelect_model[["df"]], FarmSelect_model[["lambda"]]))
  
  # Select lambda based on the degree of freedom K
  if (K <= 10) {
    nu <- FARM_df %>%
      filter(V1 == K) %>%
      filter(V2 == min(V2)) %>%
      pull(V2)
  } else {
    nu <- FARM_df %>%
      filter(V1 == K) %>%
      filter(V2 == max(V2)) %>%
      pull(V2)
  }
  
  return(list(nu = nu))
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
