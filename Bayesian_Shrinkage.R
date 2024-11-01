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
# install.packages("lars")
library(lars)
# install.packages("RSpectra")
library(RSpectra)
library(glmnet)


# **************************************************************************** #
# ********************** BAYEASIAN SHRINKAGE PREDICTIONS ********************* #
# **************************************************************************** #  

# This code aims to replicate the analysis by Christine De Mol, Domenico Giannone and Lucrezia Reichlin
# using the paper Forecasting using a large number of predictors: Is Bayesian shrinkage a valid
# alternative to principal components?" on a different dataset on EA countries

# the code compares the perform a comparison between
# RIDGE regression
# LASSO regression
# PC regression

################################################################################

# ---- LOAD DATA ----
# load EA_data untill 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

# ---- CALL FUNCTIOINS ----
# call the Bayesian Shrinkage functions
lapply(list.files("R/functions/", full.names = TRUE), source)


# ---- SET PARAMETERS ----
# Dependendent variables to be predicted
nn <- c(1)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 15, 30, 45)  # Number of principal components
K <- rr  # Number of predictors with LASSO
INfit <- seq(0.1, 0.9, by = 0.1)  # In-sample residual variance explained by Ridge
HH <- c(4)  # Steap-ahead prediction
Jwind <- 68  # Rolling window


# ********************************* IN SAMPLE ******************************** #

# Date di inizio della valutazione out-of-sample
start_y <- 2017
start_m <- 01

DATA <- as.matrix(EAdataQ[, -1])  # Matrice dei dati: tempo in righe, variabili in colonne
series <- colnames(EAdataQ)
X <- DATA

# Dimensioni del pannello
TT <- nrow(X)
NN <- ncol(X)

# Trova l'indice di inizio per l'out-of-sample
start_sample <- which(year(EAdataQ$Time) == start_y & month(EAdataQ$Time) == start_m)
if (Jwind > start_sample) stop("La finestra mobile non può essere più grande del primo campione di valutazione")

j0 <- start_sample - Jwind + 1
x <- X[j0:start_sample, ]


# ==============================================================================
# ============================= RIDGE PENALIZATION ============================= 
# ==============================================================================

# Impostazione del parametro di penalizzazione RIDGE (richiede funzione SET_ridge)
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

# Impostazione del parametro di penalizzazione LASSO (richiede funzione SET_lasso)
nu_lasso <- list()
for (jK in seq_along(K)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      nu_lasso[[paste(h, k, sep = "_")]][jK] <- SET_lasso(x[, nn[k]], x, p, K[jK], h)
    }
  }
}
nu_lasso


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

# ---- RESCALING NU (N*T) ----

T <- nrow(x)
N <- ncol(x)
multiply <- as.numeric(T*N)

# Moltiplica ogni elemento di nu_ridge per T * N usando lapply
NT_nu_ridge <- lapply(nu_ridge, function(x) as.numeric(x) * multiply)
NT_nu_ridge


# ****************************** OUT OF SAMPLE ******************************* #

# ==============================================================================
# =============================== RIDGE PREDICTION ============================= 
# ==============================================================================

# Inizializza i contenitori per le previsioni
pred_br <- vector("list", length = TT)
RIDGE <- vector("list", length = TT)
true_r <- vector("list", length = TT)
RW_r <- vector("list", length = TT)


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
        pred_br[[j]][[jfit]]<- RIDGE_pred(x[, nn[k]], x, p, nu_ridge[[k]][[jfit]], h)
        RIDGE[[j+h]][[jfit]] <- pred_br[[j]][[jfit]][["pred"]] * const
        #variance da inserire
        
        # Calcola il valore vero da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_r[[j+h]][[jfit]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_r[[j+h]][[jfit]] <- temp * const
        
        #
      }
    }
  }
}
        
# ==============================================================================
# =============================== LASSO PREDICTION ============================= 
# ==============================================================================

# Inizializza i contenitori per le previsioni
pred_bl <- vector("list", length = TT)
LASSO <- vector("list", length = TT)
true_l <- vector("list", length = TT)
RW_l <- vector("list", length = TT)

# Esegui l'esercizio di previsione fuori campione LASSO
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (jK in seq_along(K)) {
    
    for (h in HH) {  # Ciclo su numero di passi avanti
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni ridge
      for (k in seq_along(nn)) {
        pred_bl[[j]][[jK]]<- LASSO_pred(x[, nn[k]], x, p, nu_lasso[[k]][[jK]], h)
        LASSO[[j+h]][[jK]] <- pred_bl[[j]][[jK]][["pred"]] * const
        #variance da inserire
        
        # Calcola il valore vero da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_l[[j+h]][[jK]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_l[[j+h]][[jK]] <- temp * const
        
      }
    }
  }
}

# ==============================================================================
# ====================== PRINCIPAL COMPONENT PREDICTION ======================== 
# ==============================================================================

# Inizializza i contenitori per le previsioni
pred_bpc <- vector("list", length = TT)
PC <- vector("list", length = TT)
true_pc <- vector("list", length = TT)
RW_pc <- vector("list", length = TT)


# Esegui l'esercizio di previsione fuori campione
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  
  for (h in HH) {  # Ciclo su numero di passi avanti
    
    # Costanti di normalizzazione (capire l'if-else da loro)
    const <- 4
    
    # calcolo delle previsioni della PCR
    for (jr in seq_along(rr)) {
      for (k in seq_along(nn)) {
        pred_bpc[[j]][[jr]]<- PC_pred(x[, nn[k]], x, p, rr[jr], h)
        PC[[j+h]][[jr]] <- pred_bpc[[j]][[jr]][["pred"]] * const
        
        # Calcola il valore vero da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_pc[[j+h]][[jr]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_pc[[j+h]][[jr]] <- temp * const
        
      }
    }
  }  
}


# ============================== MODEL COMPARISON ==============================


        
# ****************************** EVALUATION SAMPLE **************************** #

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


# ************************ RIDGE MSFE AND VISUALIZATION ************************

# Ridge Output Matrix
RIDGE_output <- do.call(rbind, lapply(RIDGE[ind_first:length(RIDGE)], unlist))

# Rename rows and columns
rownames(RIDGE_output) <- ind_first:ind_last
colnames(RIDGE_output) <- paste0(INfit)

# True Output Matrix
true_output <- do.call(rbind, lapply(true_r[ind_first:length(true_r)], unlist))

# Rename rows and columns
rownames(true_output) <- ind_first:ind_last
colnames(true_output) <- paste0(INfit)

# MSFE ridge
MSFE_RIDGE <- matrix(colMeans((RIDGE_output - true_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_RIDGE) <- paste(INfit)
rownames(MSFE_RIDGE)<- paste(nn)

# Convertire la matrice in un dataframe
df_MSFE_RIDGE <- as.data.frame(MSFE_RIDGE)

# Convertire il dataframe in formato long
df_MSFE_RIDGE_long <- df_MSFE_RIDGE %>%
  pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

# Plot MSFE against In-sample residual variance
ggplot(df_MSFE_RIDGE_long, aes(x = Column, y = Value, group = 1)) +
  geom_line() +  # Linea per i valori
  geom_point() +  # Aggiungi punti per i valori
  labs(title = "MSFE Ridge",
       x = "In-sample residual variance",
       y = "MSFE") +
  theme_minimal()


# ************************ LASSO MSFE AND VISUALIZATION ************************

# Ridge Output Matrix
LASSO_output <- do.call(rbind, lapply(LASSO[ind_first:length(LASSO)], unlist))

# Rename rows and columns
rownames(LASSO_output) <- ind_first:ind_last
colnames(LASSO_output) <- paste0(K)

# True Output Matrix
true_l_output <- do.call(rbind, lapply(true_l[ind_first:length(true_l)], unlist))

# Rename rows and columns
rownames(true_l_output) <- ind_first:ind_last
colnames(true_l_output) <- paste0(K)

# MSFE ridge
MSFE_LASSO <- matrix(colMeans((LASSO_output - true_l_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_LASSO) <- paste(K)
rownames(MSFE_LASSO)<- paste(nn)

# Convertire la matrice in un dataframe
df_MSFE_LASSO <- as.data.frame(MSFE_LASSO)
df_MSFE_LASSO

# Convertire la matrice in un dataframe
df_MSFE_LASSO <- as.data.frame(MSFE_LASSO)

# Convertire il dataframe in formato long
df_MSFE_LASSO_long <- df_MSFE_LASSO %>%
  pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

df_MSFE_LASSO_long$Column <- as.numeric(as.character(df_MSFE_LASSO_long$Column))
df_MSFE_LASSO_long$Column <- factor(df_MSFE_LASSO_long$Column, levels = c(1, 3, 5, 10, 25, 50, 65))

# Plot MSFE against In-sample residual variance
ggplot(df_MSFE_LASSO_long, aes(x = Column, y = Value, group = 1)) +
  geom_line() +  # Linea per i valori
  geom_point() +  # Aggiungi punti per i valori
  labs(title = "MSFE LASSO",
       x = "number of predictors with non zero coefficient",
       y = "MSFE") +
  theme_minimal()




# *************************** PC MSFE AND VISUALIZATION ************************

# Ridge Output Matrix
PC_output <- do.call(rbind, lapply(PC[ind_first:length(PC)], unlist))

# Rename rows and columns
rownames(PC_output) <- ind_first:ind_last
colnames(PC_output) <- paste0(K)

# True Output Matrix
true_pc_output <- do.call(rbind, lapply(true_pc[ind_first:length(true_pc)], unlist))

# Rename rows and columns
rownames(true_pc_output) <- ind_first:ind_last
colnames(true_pc_output) <- paste0(K)

# MSFE ridge
MSFE_PC <- matrix(colMeans((PC_output - true_pc_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_PC) <- paste(K)
rownames(MSFE_PC)<- paste(nn)

# Convertire la matrice in un dataframe
df_MSFE_PC <- as.data.frame(MSFE_PC)
df_MSFE_PC

# Convertire la matrice in un dataframe
df_MSFE_PC <- as.data.frame(MSFE_PC)

# Convertire il dataframe in formato long
df_MSFE_PC_long <- df_MSFE_PC %>%
  pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

df_MSFE_PC_long$Column <- as.numeric(as.character(df_MSFE_PC_long$Column))
df_MSFE_PC_long$Column <- factor(df_MSFE_PC_long$Column, levels = c(1, 3, 5, 10, 25, 50, 65))

# Plot MSFE against In-sample residual variance
ggplot(df_MSFE_PC_long, aes(x = Column, y = Value, group = 1)) +
  geom_line() +  # Linea per i valori
  geom_point() +  # Aggiungi punti per i valori
  labs(title = "MSFE LASSO",
       x = "number of predictors with non zero coefficient",
       y = "MSFE") +
  theme_minimal()


# ==============================================================================
# ============================== MODEL COMPARISON ==============================
# ==============================================================================



# *************************** RIDGE MFSE vs. RW MSFE ***************************

# Ridge Output Matrix
RW_r_output <- do.call(rbind, lapply(RW_r[ind_first:length(RIDGE)], unlist))

# Rename rows and columns
rownames(RW_r_output) <- ind_first:ind_last
colnames(RW_r_output) <- paste0(INfit)

# MSFE RW
MSFE_r_RW <- matrix(colMeans((RW_r_output - true_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_r_RW) <- paste(INfit)
rownames(MSFE_r_RW)<- paste(nn)

# MSFE RIDGE/RW
MSFE_RIDGE_ratio <- MSFE_RIDGE / MSFE_r_RW

# *************************** LASSO MFSE vs. RW MSFE ***************************
# Ridge Output Matrix
RW_l_output <- do.call(rbind, lapply(RW_l[ind_first:length(RIDGE)], unlist))

# Rename rows and columns
rownames(RW_l_output) <- ind_first:ind_last
colnames(RW_l_output) <- paste0(K)

# MSFE RW
MSFE_l_RW <- matrix(colMeans((RW_l_output - true_l_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_l_RW) <- paste(K)
rownames(MSFE_l_RW)<- paste(nn)
MSFE_LASSO_ratio <- MSFE_LASSO / MSFE_l_RW


# ***************************** PCR MFSE vs. RW MSFE ***************************

MSFE_PC_ratio <- MSFE_PC / MSFE_l_RW









# ******************************************************************************

# Compute the MSFE for PC regression
for (jr in 1:length(rr)) {
  for (h in HH) {
    for (k in 1:length(nn)) {
      MSFE_PC[h, k, jr] <- mean((true[[h]][[k]][ind_first:ind_last, ] - PC[[h]][[k]][ind_first:ind_last, jr])^2)
      VAR_PC[h, k, jr]  <- (sd(PC[[h]][[k]][ind_first:ind_last, jr])^2) / (sd(true[[h]][[k]][ind_first:ind_last, ])^2)
    }
  }
}

# Compute the MSFE for 'naive' forecasts
for (h in HH) {
  for (k in 1:length(nn)) {
    MSFE_RW[h, k] <- mean((true[[h]][[k]][ind_first:ind_last, ] - RW[[h]][[k]][ind_first:ind_last, ])^2)
    VAR_RW[h, k]  <- (sd(RW[[h]][[k]][ind_first:ind_last, ])^2) / (sd(true[[h]][[k]][ind_first:ind_last, ])^2)
  }
}

# Compute correlation of Lasso and PC forecasts
for (jr in 1:length(rr)) {
  for (jK in 1:length(K)) {
    for (h in HH) {
      for (k in 1:length(nn)) {
        temp <- cor(PC[[h]][[k]][ind_first:ind_last, jr], LASSO[[h]][[k]][ind_first:ind_last, jK])
        R_LASSO[[h]][[k]][jK, jr] <- temp
      }
    }
  }
}

# Compute correlation of RIDGE and PC forecasts
for (jr in 1:length(rr)) {
  for (jfit in 1:length(INfit)) {
    for (h in HH) {
      for (k in 1:length(nn)) {
        temp <- cor(PC[[h]][[k]][ind_first:ind_last, jr], RIDGE[[h]][[k]][ind_first:ind_last, jfit])
        R_RIDGE[[h]][[k]][jfit, jr] <- temp
      }
    }
  }
}
