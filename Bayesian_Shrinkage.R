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
library(lubridate)  # Per la gestione delle date


# ---- LOAD THE DATA (till Covid-19) ----
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_LT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

# ---- RIDGE AND LASSO FUNCTIONS ----

# RIDGE
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


# LASSO

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

# ---- SET LASSO 
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



# ==============================================================================

# Imposta le variabili per la previsione
nn <- c(1)

# Imposta i parametri
p <- 0  # Numero di predittori laggati
rr <- c(1, 3, 5, 10, 25, 50, 75)  # Numero di componenti principali
K <- rr  # Numero di predittori da selezionare con Lasso
INfit <- seq(0.1, 0.9, by = 0.1)  # Proporzione di fit in-sample spiegata da ridge
HH <- c(4)  # Orizzonte di previsione
Jwind <- 68  # Numero di punti temporali per la stima dei parametri (2000-01-01 prende i tre mesi sucessivi)

# Date di inizio della valutazione out-of-sample
start_y <- 2017
start_m <- 01
start_d <- 01

DATA <- as.matrix(EAdataQ[, -1])  # Matrice dei dati: tempo in righe, variabili in colonne
series <- colnames(EAdataQ)
X <- DATA 

# Dimensioni del pannello
TT <- nrow(X)
NN <- ncol(X)

# Trova l'indice di inizio per l'out-of-sample
start_sample <- which(year(EAdataQ$Time) == start_y & month(EAdataQ$Time) == start_m & day(EAdataQ$Time) == start_d)
if (Jwind > start_sample) stop("La finestra mobile non può essere più grande del primo campione di valutazione")

cat("\nIl programma sta impostando i parametri di penalizzazione...\n\n")

j0 <- start_sample - Jwind + 1
x <- X[j0:start_sample, ]

# Gestione degli outlier (funzione personalizzata richiesta per sostituire outliers in R)
# x <- apply(x_temp, 2, function(col) {
  # Implementa la logica per gestire gli outlier nella serie
  # return(col)  # Da sostituire con la gestione effettiva degli outlier
# })

# x[, nn] <- x_temp[, nn]

# Impostazione del parametro di penalizzazione RIDGE (richiede funzione SET_ridge)
nu_ridge <- list()
for (jfit in seq_along(INfit)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      nu_ridge[[paste(h, k, sep = "_")]][jfit] <- SET_ridge(x[, nn[k]], x, p, INfit[jfit], h)
    }
  }
}
# varianza non spiegata dai residui : con una varianza dei residui in sample di 0.1 significa che il modello
# spiega quasi tutto e per ottenere quello la penalizzazione deve essere bassa (OLS).
# con una varianza dei residui in sample di 0.9 il modello non spiega quasi nulla quindi
# per ottenere quel livello il lambda dovrà penalizzare molto ed il modello sarà un random walk

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


T <- nrow(x)
N <- ncol(x)
multiply <- as.numeric(T*N)

# Moltiplica ogni elemento di nu_ridge per T * N usando lapply
NT_nu_ridge <- lapply(nu_ridge, function(x) as.numeric(x) * multiply)
NT_nu_ridge

# OUT OF SAMPLE RIDGE

# Inizializza i contenitori per le previsioni
#LASSO <- vector("list", length(HH))
RIDGE <- vector("list", length = TT)
#PC <- vector("list", length(HH))
true <- vector("list", length = TT)
#RW <- vector("list", length(HH))
#BETA <- vector("list", length(HH))
#INFIT <- vector("list", length(HH))
pred_b <- vector("list", length = TT)


# Esegui l'esercizio di previsione fuori campione
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (jfit in seq_along(INfit)) { 
    
    for (h in HH) {  # Ciclo su numero di passi avanti
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni ridge
      for (k in seq_along(nn)) {
        pred_b[[j]][[jfit]]<- RIDGE_pred(x[, nn[k]], x, p, nu_ridge[[k]][[jfit]], h)
        RIDGE[[j+h]][[jfit]] <- pred_b[[j]][[jfit]][["pred"]] * const
        #variance da inserire
        
        # Calcola il valore vero da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true[[j+h]][[jfit]] <- temp * const
      }
    }
  }
}

# Dates of the beginning of the evaluation sample
first_y <- 2018
first_m <- 1

# Find the index for the first out-of-sample period
ind_first <- which(year(EAdataQ$Time) == first_y & month(EAdataQ$Time) == first_m)

# Dates of the end of the evaluation sample
last_y <- 2019
last_m <- 10

# Find the index for the last out-of-sample period
ind_last <- which(year(EAdataQ$Time) == last_y & month(EAdataQ$Time) == last_m)


# Creating Array 
MSFE_RIDGE <- array(NA, dim = c(length(nn), length(INfit)))

# Compute the MSFE and Variance for RIDGE
for (jfit in seq_along(INfit)) {
  for (j in ind_first:ind_last) {
    for (h in HH) {
      for (k in seq_along(nn)) {
        # Access the true values and predicted values
        true_values <- true[[j]][[jfit]]
        ridge_values <- RIDGE[[j]][[jfit]]
        
        # Compute MSFE
        MSFE_RIDGE[k, jfit] <- mean((true_values - ridge_values) ^ 2)
        
        # Compute VAR
        #VAR_RIDGE[h, k, jfit] <- (sd(ridge_values)^2) / (sd(true_values)^2)
      }
    }
  }
}

# Crea delle matrici dagli output di RIDGE e sottrai element by element al true value per trovare l'MSE

# Ridge Output Matrix 
Ridge_output_list <- RIDGE[72:length(RIDGE)]
Ridge_output_matrix <- do.call(rbind, lapply(Ridge_output_list, function(x) unlist(x)))
rownames(Ridge_output_matrix) <- 72:(72 + length(Ridge_output_list) - 1)

col_names <- paste0("nu = ", seq(0.1, 0.9, by = 0.1))
colnames(Ridge_output_matrix) <- col_names

print(Ridge_output_matrix)

# True Output Matrix 
True_output_list <- true[72:length(true)]
True_output_matrix <- do.call(rbind, lapply(True_output_list, function(x) unlist(x)))
rownames(True_output_matrix) <- 72:(72 + length(True_output_list) - 1)

col_names <- paste0("nu = ", seq(0.1, 0.9, by = 0.1))
colnames(True_output_matrix) <- col_names

print(True_output_matrix)

diff_matrix <- Ridge_output_matrix - True_output_matrix
diff_matrix_sqr <- diff_matrix^2
MSE_RIDGE <- colMeans(diff_matrix_sqr)

# Creare una matrice con 3 righe e 3 colonne
MSFE_RIDGE <- matrix(MSE_RIDGE, nrow = 1, byrow = FALSE)  # Imposta byrow = TRUE per riempire per righe
print(MSFE_RIDGE)


print(MSE_RIDGE)

# Convertire la matrice in un dataframe long
diff_matrix_long <- as.data.frame(diff_matrix) %>%
  mutate(Row = rownames(diff_matrix)) %>%
  pivot_longer(cols = -Row, names_to = "Column", values_to = "Value")

# Creare il grafico
ggplot(diff_matrix_long, aes(x = Column, y = Value, group = Row, color = Row)) +
  geom_line() +  # Linee per ogni riga
  geom_point() +  # Aggiungi punti per i valori
  labs(title = "MSFE for each prediction", 
       x = "Colonne", 
       y = "Valori") +
  theme_minimal()  # Tema grafico

# Creazione di un esempio di MSFE_RIDGE con 9 colonne
colnames(MSFE_RIDGE) <- paste("nu", seq(0.1, 0.9, by = 0.1), sep = "")

# Convertire la matrice in un dataframe
df <- as.data.frame(MSFE_RIDGE)

# Convertire il dataframe in formato long
df_long <- df %>%
  pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

# Creare il grafico
ggplot(df_long, aes(x = Column, y = Value, group = 1)) +
  geom_line() +  # Linea per i valori
  geom_point() +  # Aggiungi punti per i valori
  labs(title = "MSFE Ridge", 
       x = "Colonne", 
       y = "MSFE") +
  theme_minimal()  # Tema grafico
