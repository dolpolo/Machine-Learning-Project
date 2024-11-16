getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project/R/functions"
setwd(path)

# ==============================================================================
# ============================ FARMSELECT FUNCTIONS ============================
# ==============================================================================


# This section will provide the function to perform model selection through a 
# Factor adjusted regularized method. 


## FarmSelect

FarmSelect_pred <- function(y, x, p, nu, h, MSFE_PC_matrix, target_index) {
  
  optimal_pc <- which.min(MSFE_PC_matrix[target_index, ])
  
  # Inizializza la matrice dei predittori con ritardi
  X <- do.call(cbind, lapply(0:p, function(j) x[(p + 1 - j):(nrow(x) - j), ]))
  
  # Standardizza la matrice dei predittori
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  # Adatta la variabile dipendente y per abbinare i predittori ritardati
  y <- y[(p + 1):length(y)]
  
  # Calcola la variabile dipendente con previsione: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)[(h + 1):length(y)]
  
  # Standardizza la variabile dipendente
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # Analisi PCA sui predittori standardizzati
  pca_res <- prcomp(XX, center = FALSE)
  
  # Ottieni i punteggi e i caricamenti dei componenti principali
  pc_scores <- pca_res$x[, 1:optimal_pc]
  pc_loadings <- pca_res$rotation[, 1:optimal_pc]
  
  # Calcola la componente ad alta correlazione
  high_corr <- pc_scores %*% t(pc_loadings)
  
  # Rimuovi la componente ad alta correlazione per ottenere predittori a bassa correlazione
  low_corr <- XX - high_corr
  
  # Prepara i predittori per la regressione
  Z <- low_corr[1:(nrow(low_corr) - h), ]
  
  # Fitta il modello Lasso
  FarmSelect_model <- glmnet(Z, y_std, alpha = 1, standardize = FALSE, lambda = nu)
  
  # Effettua le previsioni e ritorna alla scala originale
  pred <- predict(FarmSelect_model, newx = matrix(low_corr[nrow(low_corr), ], nrow = 1)) * sy + my
  
  # Identifica le variabili selezionate (coefficenti diversi da zero)
  selected_variables <- rownames(FarmSelect_model$beta)[FarmSelect_model$beta[, 1] != 0]
  
  return(list(pred = pred, model_selection = selected_variables, optimal_pc = optimal_pc))
}


## SET_FarmSelect 


SET_FarmSelect <- function(y, x, p, K, h, MSFE_PC_matrix, target_index) {
  # Trova il numero ottimale di componenti principali da rimuovere
  optimal_pc <- which.min(MSFE_PC_matrix[target_index, ])
  
  # Inizializza la matrice dei predittori con ritardi
  X <- do.call(cbind, lapply(0:p, function(j) x[(p + 1 - j):(nrow(x) - j), ]))
  
  # Standardizza la matrice dei predittori
  T <- nrow(X)
  N <- ncol(X)
  XX <- scale(X) / sqrt(N * T)
  
  # Adatta la variabile dipendente y per abbinare i predittori ritardati
  y <- y[(p + 1):length(y)]
  
  # Calcola la variabile dipendente con previsione: Y = (y_{+1} + ... + y_{+h}) / h
  Y <- stats::filter(y, rep(1/h, h), sides = 1)[(h + 1):length(y)]
  
  # Standardizza la variabile dipendente
  my <- mean(Y, na.rm = TRUE)
  sy <- sd(Y, na.rm = TRUE)
  y_std <- (Y - my) / sy
  
  # Analisi PCA sui predittori standardizzati
  pca_res <- prcomp(XX, center = FALSE)
  
  # Ottieni i punteggi e i caricamenti per il numero ottimale di componenti principali
  pc_scores <- pca_res$x[, 1:optimal_pc]
  pc_loadings <- pca_res$rotation[, 1:optimal_pc]
  
  # Calcola la componente ad alta correlazione
  high_corr <- pc_scores %*% t(pc_loadings)
  
  # Rimuovi la componente ad alta correlazione per ottenere predittori a bassa correlazione
  low_corr <- XX - high_corr
  Z <- low_corr[1:(nrow(low_corr) - h), ]
  
  # Fitta il modello Lasso con un ampio range di lambda
  FarmSelect_model <- glmnet(Z, y_std, alpha = 1, standardize = FALSE, nlambda = 2500, lambda.min.ratio = 0.00001)
  
  # Estrai i gradi di libertà e i valori di lambda
  FARM_df <- as.data.frame(cbind(FarmSelect_model[["df"]], FarmSelect_model[["lambda"]]))
  
  # Condizioni per calcolare `nu`
  if (K <= 10) {
    # Controlla che FARM_df non sia vuoto
    if (nrow(FARM_df %>% filter(V1 == K)) > 0) {
      nu <- FARM_df %>%
        filter(V1 == K) %>%
        filter(V2 == min(V2, na.rm = TRUE)) %>%
        pull(V2)
    } else {
      nu <- NA  # O un valore predefinito
    }
  } else {
    if (nrow(FARM_df %>% filter(V1 == K)) > 0) {
      nu <- FARM_df %>%
        filter(V1 == K) %>%
        filter(V2 == max(V2, na.rm = TRUE)) %>%
        pull(V2)
    } else {
      nu <- NA  # O un valore predefinito
    }
  }
  
  return(list(nu = nu))
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
convert_to_named_matrix <- function(msfe_list, target_var, col_prefix) {
  vector_lengths <- sapply(msfe_list, length)
  
  # Check if all vectors have the same length
  if (length(unique(vector_lengths)) == 1) {
    msfe_matrix <- do.call(rbind, msfe_list)
    msfe_matrix <- as.data.frame(msfe_matrix)
    
    # Set row and column names
    rownames(msfe_matrix) <- target_var
    colnames(msfe_matrix) <- paste0(col_prefix)
    
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
