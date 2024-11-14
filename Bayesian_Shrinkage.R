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
# load EA_data untill 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

# ---- CALL FUNCTIOINS ----
# call the Bayesian Shrinkage functions
# lapply(list.files("R/functions/", full.names = TRUE), source)
source("R/functions/Bayesian_shrinkage_functions.R")

# ---- SET PARAMETERS ----
# Dependendent variables to be predicted
nn <- c(1,33,97)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 25, 45, 60)  # Number of principal components
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

# target variables
target_var <- colnames(X)[nn]

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

# Impostazione del parametro di penalizzazione LASSO (richiede funzione SET_LASSO)
nu_LASSO <- list()
for (jK in seq_along(K)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      nu_LASSO[[paste(h, k, sep = "_")]][jK] <- SET_lasso(x[, nn[k]], x, p, K[jK], h)
    }
  }
}
nu_LASSO


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

# Define lengths for each level
outer_length <- TT
mid_length <- length(nn)
inner_length <- length(K)

# Initialize each list using the function
pred_br <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RIDGE <- initialize_nested_list(outer_length, c(mid_length, inner_length))
true_r <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RW_r <- initialize_nested_list(outer_length, c(mid_length, inner_length))

# Esegui l'esercizio di previsione fuori campione LASSO
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (k in seq_along(nn)) {
    
    for (jfit in seq_along(INfit)) {
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni LASSO
      for (h in HH) {  # Ciclo su numero di passi avanti
        pred_br[[j]][[k]][[jfit]]<- RIDGE_pred(x[, nn[k]], x, p, nu_ridge[[k]][[jfit]], h)
        RIDGE[[j+h]][[k]][[jfit]] <- pred_br[[j]][[k]][[jfit]][["pred"]] * const
        #variance da inserire
        
        
        # Calcola il valore vero da prevedere
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
# Inizializza i contenitori per le previsioni
# pred_br <- vector("list", length = TT)
# RIDGE <- vector("list", length = TT)
# true_r <- vector("list", length = TT)
# RW_r <- vector("list", length = TT)


# Esegui l'esercizio di previsione fuori campione
# for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
#  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
#  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
#  for (jfit in seq_along(INfit)) {
    
#    for (h in HH) {  # Ciclo su numero di passi avanti
      
      # Costanti di normalizzazione (capire l'if-else da loro)
#      const <- 4
      
      # Calcolo delle previsioni ridge
#      for (k in seq_along(nn)) {
#        pred_br[[j]][[jfit]]<- RIDGE_pred(x[, nn[k]], x, p, nu_ridge[[k]][[jfit]], h)
#        RIDGE[[j+h]][[jfit]] <- pred_br[[j]][[jfit]][["pred"]] * const
        #variance da inserire
        
        # Calcola il valore vero da prevedere
#        temp <- mean(X[(j + 1):(j + h), nn[k]])
#        true_r[[j+h]][[jfit]] <- temp * const
        
        # Constant Growth rate
#        temp <- RW_pred(x[, nn[k]], h)
#        RW_r[[j+h]][[jfit]] <- temp * const
        
        #
#      }
#    }
#  }
#}
        
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


# Esegui l'esercizio di previsione fuori campione LASSO
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (k in seq_along(nn)) {
    
    for (jK in seq_along(K)) {
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni LASSO
      for (h in HH) {  # Ciclo su numero di passi avanti
        pred_bl[[j]][[k]][[jK]]<- LASSO_pred(x[, nn[k]], x, p, nu_LASSO[[k]][[jK]], h)
        LASSO[[j+h]][[k]][[jK]] <- pred_bl[[j]][[k]][[jK]][["pred"]] * const
        #variance da inserire
        
        
        # Calcola il valore vero da prevedere
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


# Esegui l'esercizio di previsione fuori campione LASSO
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (k in seq_along(nn)) {
    
    for (jr in seq_along(rr)) {
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni PC
      for (h in HH) {  # Ciclo su numero di passi avanti
        pred_bpc[[j]][[k]][[jr]]<- PC_pred(x[, nn[k]], x, p, rr[jr], h)
        PC[[j+h]][[k]][[jr]] <- pred_bpc[[j]][[k]][[jr]][["pred"]] * const
        #variance da inserire
        
        
        # Calcola il valore vero da prevedere
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
# Inizializza i contenitori per le previsioni
# pred_bpc <- vector("list", length = TT)
# PC <- vector("list", length = TT)
# true_pc <- vector("list", length = TT)
# RW_pc <- vector("list", length = TT)


# Esegui l'esercizio di previsione fuori campione
# for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
#  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
#  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  
#  for (h in HH) {  # Ciclo su numero di passi avanti
    
    # Costanti di normalizzazione (capire l'if-else da loro)
#    const <- 4
    
    # calcolo delle previsioni della PCR
#    for (jr in seq_along(rr)) {
#      for (k in seq_along(nn)) {
#        pred_bpc[[j]][[jr]]<- PC_pred(x[, nn[k]], x, p, rr[jr], h)
#        PC[[j+h]][[jr]] <- pred_bpc[[j]][[jr]][["pred"]] * const
        
        # Calcola il valore vero da prevedere
#        temp <- mean(X[(j + 1):(j + h), nn[k]])
#        true_pc[[j+h]][[jr]] <- temp * const
        
        # Constant Growth rate
#        temp <- RW_pred(x[, nn[k]], h)
#        RW_pc[[j+h]][[jr]] <- temp * const
        
#      }
#    }
#  }  
#}


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

# Convertire il dataframe in formato long, includendo un identificatore per le righe
MSFE_RIDGE_matrix_long <- MSFE_RIDGE_matrix %>%
  rownames_to_column(var = "Variable") %>%
  pivot_longer(cols = -Variable, names_to = "Penalization", values_to = "Value")

ggplot(MSFE_RIDGE_matrix_long, aes(x = Penalization, y = Value, color = Variable, group = Variable)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  labs(
    title = "MSFE RIDGE",
    x = "In-sample variances",
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



# Ridge Output Matrix
# RIDGE_output <- do.call(rbind, lapply(RIDGE[ind_first:length(RIDGE)], unlist))

# Rename rows and columns
# rownames(RIDGE_output) <- ind_first:ind_last
# colnames(RIDGE_output) <- paste0(INfit)

# True Output Matrix
# true_output <- do.call(rbind, lapply(true_r[ind_first:length(true_r)], unlist))

# Rename rows and columns
# rownames(true_output) <- ind_first:ind_last
# colnames(true_output) <- paste0(INfit)

# MSFE ridge
# MSFE_RIDGE <- matrix(colMeans((RIDGE_output - true_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
# colnames(MSFE_RIDGE) <- paste(INfit)
# rownames(MSFE_RIDGE)<- paste(nn)

# Convertire la matrice in un dataframe
# df_MSFE_RIDGE <- as.data.frame(MSFE_RIDGE)

# Convertire il dataframe in formato long
# df_MSFE_RIDGE_long <- df_MSFE_RIDGE %>%
# pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

# Plot MSFE against In-sample residual variance
# ggplot(df_MSFE_RIDGE_long, aes(x = Column, y = Value, group = 1)) +
#  geom_line() +  # Linea per i valori
#  geom_point() +  # Aggiungi punti per i valori
#  labs(title = "MSFE Ridge",
#       x = "In-sample residual variance",
#       y = "MSFE") +
#  theme_minimal()


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

# Convertire il dataframe in formato long, includendo un identificatore per le righe
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

# Convertire il dataframe in formato long, includendo un identificatore per le righe
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


# Ridge Output Matrix
# PC_output <- do.call(rbind, lapply(PC[ind_first:length(PC)], unlist))

# Rename rows and columns
# rownames(PC_output) <- ind_first:ind_last
# colnames(PC_output) <- paste0(K)

# True Output Matrix
# true_pc_output <- do.call(rbind, lapply(true_pc[ind_first:length(true_pc)], unlist))

# Rename rows and columns
# rownames(true_pc_output) <- ind_first:ind_last
# colnames(true_pc_output) <- paste0(K)

# MSFE ridge
# MSFE_PC <- matrix(colMeans((PC_output - true_pc_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
# colnames(MSFE_PC) <- paste(K)
# rownames(MSFE_PC)<- paste(nn)

# Convertire la matrice in un dataframe
# df_MSFE_PC <- as.data.frame(MSFE_PC)
# df_MSFE_PC

# Convertire la matrice in un dataframe
# df_MSFE_PC <- as.data.frame(MSFE_PC)

# Convertire il dataframe in formato long
# df_MSFE_PC_long <- df_MSFE_PC %>%
#  pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

# df_MSFE_PC_long$Column <- as.numeric(as.character(df_MSFE_PC_long$Column))
# df_MSFE_PC_long$Column <- factor(df_MSFE_PC_long$Column, levels = c(1, 3, 5, 10, 25, 50, 65))

# Plot MSFE against In-sample residual variance
# ggplot(df_MSFE_PC_long, aes(x = Column, y = Value, group = 1)) +
#   geom_line() +  # Linea per i valori
#   geom_point() +  # Aggiungi punti per i valori
#   labs(title = "MSFE PC",
#        x = "number of predictors with non zero coefficient",
#        y = "MSFE") +
#   theme_minimal()


# ==============================================================================
# ============================== MODEL COMPARISON ==============================
# ==============================================================================



# *************************** RIDGE MlE vs. RW MSFE ***************************

# Ridge Output Matrix
# RW_r_output <- do.call(rbind, lapply(RW_r[ind_first:length(RIDGE)], unlist))

# Rename rows and columns
# rownames(RW_r_output) <- ind_first:ind_last
# colnames(RW_r_output) <- paste0(INfit)

# MSFE RW
# MSFE_r_RW <- matrix(colMeans((RW_r_output - true_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
# colnames(MSFE_r_RW) <- paste(INfit)
# rownames(MSFE_r_RW)<- paste(nn)

# MSFE RIDGE/RW
# MSFE_RIDGE_ratio <- MSFE_RIDGE / MSFE_r_RW

# *************************** LASSO MlE vs. RW MSFE ***************************
# Ridge Output Matrix
# RW_l_output <- do.call(rbind, lapply(RW_l[ind_first:length(LASSO)], unlist))

# Rename rows and columns
# rownames(RW_l_output) <- ind_first:ind_last
# colnames(RW_l_output) <- paste0(K)

# MSFE RW
# MSFE_l_RW <- matrix(colMeans((RW_l_output - true_l_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
# colnames(MSFE_l_RW) <- paste(K)
# rownames(MSFE_l_RW)<- paste(nn)
# MSFE_LASSO_ratio <- MSFE_LASSO / MSFE_l_RW


# ***************************** PCR MlE vs. RW MSFE ***************************

# MSFE_PC_ratio <- MSFE_PC / MSFE_l_RW


# ==============================================================================
# =================================== BEST MODEL ===============================
# ==============================================================================


# ---- RIDGE ----

# Inizializza una lista vuota per salvare i risultati
best_r_model <- data.frame(Variable = rownames(MSFE_RIDGE_matrix),
                           Best_MSFE = numeric(nrow(MSFE_RIDGE_matrix)),
                           Best_Penalization = character(nrow(MSFE_RIDGE_matrix)),
                           nu_RIDGE_r = numeric(nrow(MSFE_RIDGE_matrix)),
                           stringsAsFactors = FALSE)

# Crea una mappatura tra i nomi delle penalizzazioni e gli indici numerici
penalization_map <- c( "0.1" = 1, "0.2" = 2, "0.3" = 3, "0.4" = 4, 
                       "0.5" = 5, "0.6" = 6, "0.7" = 7, "0.8" = 8, "0.9" = 9 )

# Loop per ogni riga della matrice MSFE
for (i in 1:nrow(MSFE_RIDGE_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i
  min_index <- which.min(MSFE_RIDGE_matrix[i, ])
  
  # Assegna il valore minimo e il nome della colonna nella lista dei risultati
  best_r_model$Best_MSFE[i] <- MSFE_RIDGE_matrix[i, min_index]
  best_r_model$Best_Penalization[i] <- colnames(MSFE_RIDGE_matrix)[min_index]
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- best_r_model$Best_Penalization[i]
  
  # Controlla se la penalizzazione è presente nel mapping
  if (best_penalization %in% names(penalization_map)) {
    
    # Ottieni l'indice corrispondente alla penalizzazione
    penalization_index <- penalization_map[best_penalization]
    
    # Seleziona la predizione corrispondente dal nu_LASSO
    best_r_model$nu_RIDGE_r[i] <- nu_ridge[[i]][[penalization_index]]
    
  } else {
    # Gestisci il caso in cui la penalizzazione non viene trovata
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_RIDGE_matrix)[i]))
    best_r_model$nu_RIDGE_r[i] <- NA  # Assegna NA se la penalizzazione non è trovata
  }
}

# Visualizza i risultati
print(best_r_model)

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
best_r_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(RIDGE))), ncol = nrow(MSFE_RIDGE_matrix))
rownames(best_r_prediction) <- seq(from = ind_first, to = length(RIDGE))  # Assegna gli anni come nomi delle righe
colnames(best_r_prediction) <- rownames(MSFE_RIDGE_matrix)  # Assegna le variabili come nomi delle colonne

# Loop per ogni variabile (riga di MSFE_RIDGE_matrix)
for (i in 1:nrow(MSFE_RIDGE_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i (la penalizzazione ottimale)
  min_index <- which.min(MSFE_RIDGE_matrix[i, ])
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- colnames(MSFE_RIDGE_matrix)[min_index]
  
  # Ottieni l'indice della penalizzazione corrispondente
  penalization_index <- penalization_map[best_penalization]
  
  # Per ogni anno (da ind_first a RIDGE), seleziona la predizione corrispondente alla penalizzazione ottimale
  for (j in ind_first:length(RIDGE)) {  # Itera sugli anni
    # Estrai la lista di predizioni per la variabile i e l'anno j da RIDGE
    pred_blist_for_year_var <- RIDGE[[j]][[i]]
    
    # Controlla se penalization_index è dentro i limiti della lista di predizioni
    if (penalization_index <= length(pred_blist_for_year_var)) {
      # Salva la predizione corrispondente al miglior parametro di penalizzazione
      best_r_prediction[j - ind_first + 1, i] <- pred_blist_for_year_var[[penalization_index]]
    } else {
      # Se l'indice è fuori limite, assegna NA
      best_r_prediction[j - ind_first + 1, i] <- NA
      print(paste("Attenzione: Penalization", best_penalization, "è fuori limite per la variabile", rownames(MSFE_l_matrix)[i], "e l'anno", j))
    }
  }
}

# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(best_r_prediction)

best_r_prediction <- as.data.frame(best_r_prediction)



path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_r_model, file = "Results/Best Models/best_r_model.rds")
saveRDS(best_r_prediction, file = "Results/Best Models/best_r_prediction.rds")




# ---- LASSO ----

# Inizializza una lista vuota per salvare i risultati
best_l_model <- data.frame(Variable = rownames(MSFE_l_matrix),
                            Best_MSFE = numeric(nrow(MSFE_l_matrix)),
                            Best_Penalization = character(nrow(MSFE_l_matrix)),
                            nu_LASSO_l = numeric(nrow(MSFE_l_matrix)),
                            stringsAsFactors = FALSE)

# Crea una mappatura tra i nomi delle penalizzazioni e gli indici numerici
penalization_map <- c("1" = 1, "3" = 2, "5" = 3, "10" = 4, 
                      "25" = 5, "45" = 6, "60" = 7)

# Loop per ogni riga della matrice MSFE
for (i in 1:nrow(MSFE_l_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i
  min_index <- which.min(MSFE_l_matrix[i, ])
  
  # Assegna il valore minimo e il nome della colonna nella lista dei risultati
  best_l_model$Best_MSFE[i] <- MSFE_l_matrix[i, min_index]
  best_l_model$Best_Penalization[i] <- colnames(MSFE_l_matrix)[min_index]
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- best_l_model$Best_Penalization[i]
  
  # Controlla se la penalizzazione è presente nel mapping
  if (best_penalization %in% names(penalization_map)) {
    
    # Ottieni l'indice corrispondente alla penalizzazione
    penalization_index <- penalization_map[best_penalization]
    
    # Seleziona la predizione corrispondente dal nu_LASSO
    best_l_model$nu_LASSO_l[i] <- nu_LASSO[[i]][[penalization_index]]
    
  } else {
    # Gestisci il caso in cui la penalizzazione non viene trovata
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_l_matrix)[i]))
    best_l_model$nu_LASSO_l[i] <- NA  # Assegna NA se la penalizzazione non è trovata
  }
}

# Visualizza i risultati
print(best_l_model)

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
best_l_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(LASSO))), ncol = nrow(MSFE_l_matrix))
rownames(best_l_prediction) <- seq(from = ind_first, to = length(LASSO))  # Assegna gli anni come nomi delle righe
colnames(best_l_prediction) <- rownames(MSFE_l_matrix)  # Assegna le variabili come nomi delle colonne


# Loop per ogni variabile (riga di MSFE_l_matrix)
for (i in 1:nrow(MSFE_l_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i (la penalizzazione ottimale)
  min_index <- which.min(MSFE_l_matrix[i, ])
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- colnames(MSFE_l_matrix)[min_index]
  
  # Ottieni l'indice della penalizzazione corrispondente
  penalization_index <- penalization_map[best_penalization]
  
  # Per ogni anno (da ind_first a LASSO), seleziona la predizione corrispondente alla penalizzazione ottimale
  for (j in ind_first:length(LASSO)) {  # Itera sugli anni
    # Estrai la lista di predizioni per la variabile i e l'anno j da LASSO
    pred_blist_for_year_var <- LASSO[[j]][[i]]
    
    # Controlla se penalization_index è dentro i limiti della lista di predizioni
    if (penalization_index <= length(pred_blist_for_year_var)) {
      # Salva la predizione corrispondente al miglior parametro di penalizzazione
      best_l_prediction[j - ind_first + 1, i] <- pred_blist_for_year_var[[penalization_index]]
    } else {
      # Se l'indice è fuori limite, assegna NA
      best_l_prediction[j - ind_first + 1, i] <- NA
      print(paste("Attenzione: Penalization", best_penalization, "è fuori limite per la variabile", rownames(MSFE_l_matrix)[i], "e l'anno", j))
    }
  }
}

# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(best_l_prediction)

best_l_prediction <- as.data.frame(best_l_prediction)


path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_l_model, file = "Results/Best Models/best_l_model.rds")
saveRDS(best_l_prediction, file = "Results/Best Models/best_l_prediction.rds")


# ---- PC ---- 

# Inizializza una lista vuota per salvare i risultati
best_PC_model <- data.frame(Variable = rownames(MSFE_PC_matrix),
                            Best_MSFE = numeric(nrow(MSFE_PC_matrix)),
                            Best_Parameter = character(nrow(MSFE_PC_matrix)),
                            stringsAsFactors = FALSE)

# Crea una mappatura tra i nomi delle penalizzazioni e gli indici numerici
penalization_map <- c("1" = 1, "3" = 2, "5" = 3, "10" = 4, 
                      "25" = 5, "45" = 6, "60" = 7)

# Loop per ogni riga della matrice MSFE
for (i in 1:nrow(MSFE_PC_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i
  min_index <- which.min(MSFE_PC_matrix[i, ])
  
  # Assegna il valore minimo e il nome della colonna nella lista dei risultati
  best_PC_model$Best_MSFE[i] <- MSFE_PC_matrix[i, min_index]
  best_PC_model$Best_Parameter[i] <- colnames(MSFE_PC_matrix)[min_index]
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_parameter <- best_PC_model$Best_Parameter[i]
  
  # Controlla se la penalizzazione è presente nel mapping
  if (best_parameter %in% names(penalization_map)) {
    
    # Ottieni l'indice corrispondente alla penalizzazione
    penalization_index <- penalization_map[best_parameter]
    
  } else {
    # Gestisci il caso in cui la penalizzazione non viene trovata
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_l_matrix)[i]))
    best_l_model$nu_LASSO_l[i] <- NA  # Assegna NA se la penalizzazione non è trovata
  }
}

# Visualizza i risultati
print(best_l_model)

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
best_pc_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(PC))), ncol = nrow(MSFE_PC_matrix))
rownames(best_pc_prediction) <- seq(from = ind_first, to = length(PC))  # Assegna gli anni come nomi delle righe
colnames(best_pc_prediction) <- rownames(MSFE_PC_matrix)  # Assegna le variabili come nomi delle colonne


# Loop per ogni variabile (riga di MSFE_PC_matrix)
for (i in 1:nrow(MSFE_PC_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i (la penalizzazione ottimale)
  min_index <- which.min(MSFE_PC_matrix[i, ])
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- colnames(MSFE_PC_matrix)[min_index]
  
  # Ottieni l'indice della penalizzazione corrispondente
  penalization_index <- penalization_map[best_penalization]
  
  # Per ogni anno (da ind_first a PC), seleziona la predizione corrispondente alla penalizzazione ottimale
  for (j in ind_first:length(PC)) {  # Itera sugli anni
    # Estrai la lista di predizioni per la variabile i e l'anno j da PC
    pred_blist_for_year_var <- PC[[j]][[i]]
    
    # Controlla se penalization_index è dentro i limiti della lista di predizioni
    if (penalization_index <= length(pred_blist_for_year_var)) {
      # Salva la predizione corrispondente al miglior parametro di penalizzazione
      best_pc_prediction[j - ind_first + 1, i] <- pred_blist_for_year_var[[penalization_index]]
    } else {
      # Se l'indice è fuori limite, assegna NA
      best_pc_prediction[j - ind_first + 1, i] <- NA
      print(paste("Attenzione: Penalization", best_penalization, "è fuori limite per la variabile", rownames(MSFE_l_matrix)[i], "e l'anno", j))
    }
  }
}

# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(best_pc_prediction)

best_pc_prediction <- as.data.frame(best_pc_prediction)


path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_PC_model, file = "Results/Best Models/best_PC_model.rds")
saveRDS(best_pc_prediction, file = "Results/Best Models/best_pc_prediction.rds")


# ==============================================================================
# ================= VARIABLE FREQUENCY LASSO MODEL SELECTIN  ===================
# ==============================================================================

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
variable_selection <- matrix(NA, nrow = length(seq(from = start_sample, to = length(LASSO) - HH)), ncol = nrow(MSFE_l_matrix))
rownames(variable_selection) <- seq(from = start_sample, to = length(LASSO) - HH)  # Assegna gli anni come nomi delle righe
colnames(variable_selection) <- rownames(MSFE_l_matrix)  # Assegna le variabili come nomi delle colonne

# Loop per ogni variabile (riga di MSFE_l_matrix)
for (i in 1:nrow(MSFE_l_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i (la penalizzazione ottimale)
  min_index <- which.min(MSFE_l_matrix[i, ])
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- colnames(MSFE_l_matrix)[min_index]
  
  # Ottieni l'indice della penalizzazione corrispondente
  penalization_index <- penalization_map[best_penalization]
  
  # Per ogni anno (da start_sample a LASSO), seleziona la predizione corrispondente alla penalizzazione ottimale
  for (j in start_sample:(length(LASSO) - HH)) {  # Itera sugli anni
    # Estrai la lista di predizioni per la variabile i e l'anno j da LASSO
    pred_blist_for_year_var <- pred_bl[[j]][[i]]
    
    # Verifica che la lista di predizioni per l'anno e la variabile sia valida
    if (is.list(pred_blist_for_year_var) && length(pred_blist_for_year_var) >= penalization_index) {
      # Seleziona il modello corrispondente alla penalizzazione ottimale
      selected_model <- pred_blist_for_year_var[[penalization_index]]
      
      # Controlla se 'model_selection' è una lista (potrebbe contenere più modelli)
      if ("model_selection" %in% names(selected_model)) {
        model_selection_values <- selected_model[["model_selection"]]
        
        # Se 'model_selection' è una lista o vettore, puoi decidere come selezionare i valori
        if (length(model_selection_values) > 1) {
          # Ad esempio, prendi tutti i valori di 'model_selection' e assegnali
          # Potresti anche voler fare un'aggregazione, come la media, se vuoi un singolo valore
          variable_selection[j - start_sample + 1, i] <- paste(model_selection_values, collapse = ", ")
        } else {
          # Se c'è solo un valore, assegna quel valore
          variable_selection[j - start_sample + 1, i] <- model_selection_values
        }
      } else {
        # Se 'model_selection' non è presente nel modello, assegna NA
        variable_selection[j - start_sample + 1, i] <- NA
        print(paste("Attenzione: 'model_selection' non trovato per la variabile", rownames(MSFE_l_matrix)[i], "e l'anno", j))
      }
    } else {
      # Se l'indice è fuori limite o la lista non è valida, assegna NA
      variable_selection[j - start_sample + 1, i] <- NA
      print(paste("Attenzione: Penalization", best_penalization, "è fuori limite per la variabile", rownames(MSFE_l_matrix)[i], "e l'anno", j))
    }
  }
}


# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(variable_selection)

# the model selection is consistent. Every time we ask to select the variable according to the best penalization it extracts the same variables
# in gdp it is not the case probabluy due to the fact it is high correlated?

# Converti la matrice in formato long
variable_selection_long <- as.data.frame(variable_selection) %>%
  mutate(Year = rownames(variable_selection)) %>%
  gather(key = "Variable", value = "SelectedModel", -Year)

# Conta la frequenza delle selezioni per ogni variabile target
freq_table <- variable_selection_long %>%
  group_by(Variable, SelectedModel) %>%
  summarise(Frequency = n(), .groups = 'drop')

# Crea un grafico delle frequenze delle selezioni senza la legenda
ggplot(freq_table, aes(x = Variable, y = Frequency, fill = SelectedModel)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Frequency of Variable Selection for Each Target",
       x = "Target Variable",
       y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Ruota le etichette dell'asse x
  guides(fill = "none")  # Rimuove la legenda




# ==============================================================================
# ========================== PREDICTION VISUALIZATION ========================== 
# ==============================================================================

# time serires of gdp

# EAdataQ_long <- EAdataQ %>%
#  select(Time, GDP_EA)
# ggplot(EAdataQ_long, aes(x = Time, y = GDP_EA)) +
#  geom_line() + 
#  geom_vline(xintercept = as.Date("2017-01-01"), color = "red", linetype = "dashed", linewidth = 1) +
#labs(title = "Time series of GDP in Euro Area",
#       x = "Year", 
#       y = "GDP in Euro Area") +
#  theme_minimal() 


# Ridge_opt <- RIDGE_output[,6]
# Ridge_df <- merge(EAdataQ_long, Ridge_opt, by = "row.names")

# Crea Ridge_opt_df con la colonna chiave
# Ridge_opt_df <- data.frame(Row.names = rownames(RIDGE_output), Ridge_opt)

# Assicurati che EAdataQ_long abbia una colonna chiave per fare il join
# EAdataQ_long <- EAdataQ_long %>%
#   mutate(Row.names = rownames(EAdataQ_long))

# Ridge_opt_df <- Ridge_opt_df %>%
#  mutate(Ridge_opt = Ridge_opt/4)

# Unisci i dataset
# Ridge_df <- left_join(EAdataQ_long, Ridge_opt_df, by = "Row.names")


# Grafico con ggplot
# ggplot(Ridge_df, aes(x = Time)) +
#   geom_line(aes(y = GDP_EA), colour = "blue", linewidth = 0.5) +  # Linea completa del GDP_EA
#  geom_line(aes(y = Ridge_opt), colour = "green", linewidth = 1, na.rm = TRUE) +  # Linea di Ridge_opt solo dove disponibile
#  geom_vline(xintercept = as.Date("2017-01-01"), color = "red", linetype = "dashed", linewidth = 1) +
#  labs(title = "Time series of GDP in Euro Area",
#       x = "Year", 
#       y = "GDP in Euro Area")

################################### LASSO PREDICTION ###########################


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

# La varianza cumulata dimostra che ci sta ancora molto da spiegare dopo la prima componente. non è noice


# fai il grafico coi l'eigengap tra la prima e la seconda componente.... chissa come se fa

# Calcola l'eigengap
# eigengap <- diff(varianze)

# Crea il grafico dell'eigengap
# plot(eigengap, type = "b", pch = 19, col = "red", 
    #  xlab = "Componenti Principali (PC)", 
    #  ylab = "Eigengap", 
    #  main = "Grafico dell'Eigengap")


# Interpretations od PC 

pca_res$rotation[,c(1,2,3)]


# ==============================================================================
# ============================ PREDICTION VISUALIZATION ========================
# ==============================================================================

# ---- RIDGE ----

# Verifica che `nn` contenga i nomi delle colonne corretti
EAdataQ_long <- EAdataQ %>%
  select(Time, all_of(nn+1)) %>%  # Seleziona la colonna Time e le colonne specificate in nn
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value")

# Converti la variabile Time in formato Date per rimuovere l'informazione UTC
time_subset <- as.Date(EAdataQ$Time[ind_first:length(RIDGE)])

best_r_prediction_long <- best_r_prediction %>%
  mutate(Time = time_subset)%>%  # Aggiungi la colonna Time
  mutate(across(where(is.numeric), ~ . / HH)) %>% 
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value_pred")

output_pred_r <- left_join(EAdataQ_long, best_r_prediction_long, by = c("Time", "Variable"))

# Ristrutturazione dei dati per avere una colonna 'Type' che distingue tra 'Value' e 'Value_pred'
output_pred_r_long <- output_pred_r %>%
  pivot_longer(cols = c(Value, Value_pred), 
               names_to = "Type", 
               values_to = "Value")

ggplot(output_pred_r_long, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1.2) +  # Aumenta lo spessore delle linee per renderle più visibili
  facet_wrap(~ Variable, scales = "free_y", ncol = 1) +  # Un grafico per ogni variabile, con una colonna per facilitare la lettura
  scale_color_manual(values = c("blue", "red")) +  # Personalizza i colori per 'Value' e 'Value_pred'
  labs(title = "Time Series of Variables in the Euro Area",
       subtitle = "Actual and RIDGE Predicted Values",
       x = "Year", 
       y = "Value",
       color = "Type",
       caption = "Source: Euro Area Data") +  # Aggiungi un sottotitolo e una didascalia
  theme_minimal(base_size = 14) +  # Imposta una dimensione base per il tema
  theme(
    legend.position = "top",  # Posiziona la legenda in cima
    panel.grid.major = element_line(color = "gray", linetype = "dashed", size = 0.5),  # Aggiungi linee della griglia maggiori in grigio
    panel.grid.minor = element_blank(),  # Rimuovi la griglia minore
    strip.background = element_rect(fill = "lightblue", color = "black", size = 1),  # Colore di sfondo per le etichette dei facetti
    strip.text = element_text(size = 12, face = "bold"),  # Cambia la dimensione e il font delle etichette dei facetti
    axis.title.x = element_text(size = 14, face = "bold"),  # Titolo x
    axis.title.y = element_text(size = 14, face = "bold"),  # Titolo y
    axis.text.x = element_text(angle = 45, hjust = 1),  # Ruota le etichette dell'asse x per migliorarne la leggibilità
    axis.text.y = element_text(size = 12)  # Cambia la dimensione delle etichette dell'asse y
  ) 



# ---- LASSO ----

# Verifica che `nn` contenga i nomi delle colonne corretti
EAdataQ_long <- EAdataQ %>%
  select(Time, all_of(nn+1)) %>%  # Seleziona la colonna Time e le colonne specificate in nn
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value")

# Converti la variabile Time in formato Date per rimuovere l'informazione UTC
time_subset <- as.Date(EAdataQ$Time[ind_first:length(LASSO)])

best_l_prediction_long <- best_l_prediction %>%
  mutate(Time = time_subset)%>%  # Aggiungi la colonna Time
  mutate(across(where(is.numeric), ~ . / HH)) %>% 
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value_pred")

output_pred <- left_join(EAdataQ_long, best_l_prediction_long, by = c("Time", "Variable"))

# Ristrutturazione dei dati per avere una colonna 'Type' che distingue tra 'Value' e 'Value_pred'
output_pred_long <- output_pred %>%
  pivot_longer(cols = c(Value, Value_pred), 
               names_to = "Type", 
               values_to = "Value")

ggplot(output_pred_long, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1.2) +  # Aumenta lo spessore delle linee per renderle più visibili
  facet_wrap(~ Variable, scales = "free_y", ncol = 1) +  # Un grafico per ogni variabile, con una colonna per facilitare la lettura
  scale_color_manual(values = c("blue", "red")) +  # Personalizza i colori per 'Value' e 'Value_pred'
  labs(title = "Time Series of Variables in the Euro Area",
       subtitle = "Actual and LASSO Predicted Values",
       x = "Year", 
       y = "Value",
       color = "Type",
       caption = "Source: Euro Area Data") +  # Aggiungi un sottotitolo e una didascalia
  theme_minimal(base_size = 14) +  # Imposta una dimensione base per il tema
  theme(
    legend.position = "top",  # Posiziona la legenda in cima
    panel.grid.major = element_line(color = "gray", linetype = "dashed", size = 0.5),  # Aggiungi linee della griglia maggiori in grigio
    panel.grid.minor = element_blank(),  # Rimuovi la griglia minore
    strip.background = element_rect(fill = "lightblue", color = "black", size = 1),  # Colore di sfondo per le etichette dei facetti
    strip.text = element_text(size = 12, face = "bold"),  # Cambia la dimensione e il font delle etichette dei facetti
    axis.title.x = element_text(size = 14, face = "bold"),  # Titolo x
    axis.title.y = element_text(size = 14, face = "bold"),  # Titolo y
    axis.text.x = element_text(angle = 45, hjust = 1),  # Ruota le etichette dell'asse x per migliorarne la leggibilità
    axis.text.y = element_text(size = 12)  # Cambia la dimensione delle etichette dell'asse y
  ) 

# ---- PC ----

# Verifica che `nn` contenga i nomi delle colonne corretti
EAdataQ_long <- EAdataQ %>%
  select(Time, all_of(nn+1)) %>%  # Seleziona la colonna Time e le colonne specificate in nn
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value")

# Converti la variabile Time in formato Date per rimuovere l'informazione UTC
time_subset <- as.Date(EAdataQ$Time[ind_first:length(RIDGE)])

best_pc_prediction_long <- best_pc_prediction %>%
  mutate(Time = time_subset)%>%  # Aggiungi la colonna Time
  mutate(across(where(is.numeric), ~ . / HH)) %>% 
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value_pred")

output_pred_r <- left_join(EAdataQ_long, best_r_prediction_long, by = c("Time", "Variable"))

# Ristrutturazione dei dati per avere una colonna 'Type' che distingue tra 'Value' e 'Value_pred'
output_pred_r_long <- output_pred_r %>%
  pivot_longer(cols = c(Value, Value_pred), 
               names_to = "Type", 
               values_to = "Value")

ggplot(output_pred_r_long, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1.2) +  # Aumenta lo spessore delle linee per renderle più visibili
  facet_wrap(~ Variable, scales = "free_y", ncol = 1) +  # Un grafico per ogni variabile, con una colonna per facilitare la lettura
  scale_color_manual(values = c("blue", "red")) +  # Personalizza i colori per 'Value' e 'Value_pred'
  labs(title = "Time Series of Variables in the Euro Area",
       subtitle = "Actual and PC Predicted Values",
       x = "Year", 
       y = "Value",
       color = "Type",
       caption = "Source: Euro Area Data") +  # Aggiungi un sottotitolo e una didascalia
  theme_minimal(base_size = 14) +  # Imposta una dimensione base per il tema
  theme(
    legend.position = "top",  # Posiziona la legenda in cima
    panel.grid.major = element_line(color = "gray", linetype = "dashed", size = 0.5),  # Aggiungi linee della griglia maggiori in grigio
    panel.grid.minor = element_blank(),  # Rimuovi la griglia minore
    strip.background = element_rect(fill = "lightblue", color = "black", size = 1),  # Colore di sfondo per le etichette dei facetti
    strip.text = element_text(size = 12, face = "bold"),  # Cambia la dimensione e il font delle etichette dei facetti
    axis.title.x = element_text(size = 14, face = "bold"),  # Titolo x
    axis.title.y = element_text(size = 14, face = "bold"),  # Titolo y
    axis.text.x = element_text(angle = 45, hjust = 1),  # Ruota le etichette dell'asse x per migliorarne la leggibilità
    axis.text.y = element_text(size = 12)  # Cambia la dimensione delle etichette dell'asse y
  ) 

# ==============================================================================
# ===================== COMPARISON IN PREDICTION VISUALIZATION =================
# ==============================================================================

# In this section are presented the out of sample performances comparison between models

best_r_prediction <- best_r_prediction %>%
  rename_with(~ paste0("r_", .), everything())

best_l_prediction <- best_l_prediction %>%
  rename_with(~ paste0("l_", .), everything())

best_pc_prediction <- best_pc_prediction %>%
  rename_with(~ paste0("pc_", .), everything())

prediction_comparison_list <- list(best_r_prediction, best_l_prediction, best_pc_prediction)

# Applica la sequenza come nuova colonna "Index" per ciascun data frame nella lista
prediction_comparison_list <- lapply(prediction_comparison_list, function(df) {
  df %>%
    mutate(Index = (start_sample+HH):length(RIDGE)) 
})

# Ora puoi effettuare il merge usando "Index" come chiave
prediction_comparison <- reduce(prediction_comparison_list, left_join, by = "Index")
prediction_comparison <- prediction_comparison%>%
  select(-Index)

write_xlsx(prediction_comparison, "Results/Best Models/prediction_comparison.xlsx")

prediction_comparison_long <- prediction_comparison %>%
  mutate(Time = time_subset)%>%  # Aggiungi la colonna Time
  mutate(across(where(is.numeric), ~ . / HH)) %>% 
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value_pred")


# Espandiamo il dataset creando una riga per ogni combinazione di variabile e prefisso
EAdataQ_long_all <- EAdataQ_long %>%
  # Creiamo una griglia di combinazioni per ogni variabile con i prefissi
  expand_grid(Prefix = c("pc_", "r_", "l_")) %>%
  # Ripetiamo per ogni variabile la sequenza di prefissi
  mutate(Variable = str_c(Prefix, rep(EAdataQ_long$Variable, each = 3))) %>%
  # Ordinamento in base a 'Variable' e 'Time'
  arrange(Variable, Time) %>%
  select(-Prefix)  # Rimuoviamo la colonna 'Prefix' non più necessaria


output_pred_comp <- left_join(EAdataQ_long_all, prediction_comparison_long, by = c("Time", "Variable"))
output_pred_comp <- output_pred_comp %>%
  arrange(Time)

# Ristrutturazione dei dati per avere una colonna 'Type' che distingue tra 'Value' e 'Value_pred'
output_pred_comp <- output_pred_comp %>%
  pivot_longer(cols = c(Value, Value_pred), 
               names_to = "Type", 
               values_to = "Value") %>%
  # Se desideri differenziare per 'Variable' puoi anche assicurarti di mantenere la colonna 'Variable' intatta
  arrange(Variable, Time) 

# Create different datasets for every variable to predict

## GDP

# Make sure the 'output_pred_comp' dataset is already filtered and in long format
output_pred_comp_GDP <- output_pred_comp %>%
  filter(Variable %in% c("l_GDP_EA", "pc_GDP_EA", "r_GDP_EA"))

# Add a column to distinguish models in the predictions
output_pred_comp_GDP_long <- output_pred_comp_GDP %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "l_") ~ "LASSO",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_GDP_long, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +  # Increase line thickness
  labs(
    title = "Comparison of Models for GDP in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model Type",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",   # Position the legend at the bottom
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Center and bold title
    axis.title = element_text(size = 12),  # Increase axis titles size
    axis.text = element_text(size = 10),  # Increase axis text size
    panel.grid = element_blank(),  # Remove gridlines for a cleaner look
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)  # Add a border to the plot
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "blue", 
      "RIDGE.Value_pred" = "red", 
      "LASSO.Value_pred" = "green",
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))

## GGLB.LLN

# Make sure the 'output_pred_comp' dataset is already filtered and in long format
output_pred_comp_GGL <- output_pred_comp %>%
  filter(Variable %in% c("r_GGLB.LLN_EA", "pc_GGLB.LLN_EA", "l_GGLB.LLN_EA"))

# Add a column to distinguish models in the predictions
output_pred_comp_GGL_long <- output_pred_comp_GGL %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "l_") ~ "LASSO",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_GGL_long, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +  # Increase line thickness
  labs(
    title = "Comparison of Models for GGLB.LLN in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model Type",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",   # Position the legend at the bottom
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Center and bold title
    axis.title = element_text(size = 12),  # Increase axis titles size
    axis.text = element_text(size = 10),  # Increase axis text size
    panel.grid = element_blank(),  # Remove gridlines for a cleaner look
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)  # Add a border to the plot
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "blue", 
      "RIDGE.Value_pred" = "red", 
      "LASSO.Value_pred" = "green",
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))


## PRICES

# Make sure the 'output_pred_comp' dataset is already filtered and in long format
output_pred_comp_PP <- output_pred_comp %>%
  filter(Variable %in% c("r_PPINRG_EA", "pc_PPINRG_EA", "l_PPINRG_EA"))

# Add a column to distinguish models in the predictions
output_pred_comp_PP_long <- output_pred_comp_PP %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "l_") ~ "LASSO",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_PP_long, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +  # Increase line thickness
  labs(
    title = "Comparison of Models for Energy Prices in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model Type",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",   # Position the legend at the bottom
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Center and bold title
    axis.title = element_text(size = 12),  # Increase axis titles size
    axis.text = element_text(size = 10),  # Increase axis text size
    panel.grid = element_blank(),  # Remove gridlines for a cleaner look
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)  # Add a border to the plot
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "blue", 
      "RIDGE.Value_pred" = "red", 
      "LASSO.Value_pred" = "green",
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))


# ==============================================================================
# ============================ CORRELATION WITH PC =============================
# ==============================================================================

# In the following section it is presented the correlation among Ridge and LASSO 
# forecasts with principal component forecasts taking the best number of PC. Principal
# components and Ridge forecasts are highly correlated, particularly
# when the prior is such that the forecasting performances are good.

# RIDGE

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
# Inizializza la matrice all_ridge_prediction
all_ridge_prediction <- matrix(NA, 
                               nrow = length(seq(from = ind_first, to = length(RIDGE))), 
                               ncol = ncol(MSFE_RIDGE_matrix) * nrow(MSFE_RIDGE_matrix))

# Assegna gli anni come nomi delle righe
rownames(all_ridge_prediction) <- seq(from = ind_first, to = length(RIDGE))  

# Assegna le variabili come nomi delle colonne (qui utilizziamo un vettore per ogni variabile)
col_names <- unlist(lapply(1:nrow(MSFE_RIDGE_matrix), function(i) paste0(colnames(MSFE_RIDGE_matrix), "_var", i)))
colnames(all_ridge_prediction) <- col_names

# Loop per ogni anno e variabile
for (j in seq(from = ind_first, to = length(RIDGE))) { 
  for (i in 1:nrow(MSFE_RIDGE_matrix)) {  # Loop per ogni variabile
    for (jfit in seq_along(INfit)) {  # Loop per ogni penalizzazione jfit
      # Estrai la lista di predizioni per la variabile i e l'anno j con la penalizzazione jK
      pred_blist_for_year_var <- RIDGE[[j]][[i]][[jfit]]
      
      # Posizione della colonna corrispondente alla variabile i
      col_position <- (i - 1) * ncol(MSFE_RIDGE_matrix) + jfit
      all_ridge_prediction[j - ind_first + 1, col_position] <- pred_blist_for_year_var
    }
  }
}


# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(all_ridge_prediction)

all_ridge_prediction <- as.data.frame(all_ridge_prediction)


# use best PC prediction to compute the correlation for every variable


# Supponiamo che il numero di variabili sia pari al numero di righe in best_pc_prediction
n_variabili <- ncol(best_pc_prediction)
n_penalizzazioni <- ncol(MSFE_RIDGE_matrix)  # Numero di penalizzazioni (suppongo INfit sia definito)

# Inizializza la matrice delle correlazioni
correlation_matrix_PC_RIDGE <- matrix(NA, nrow = n_variabili, ncol = n_penalizzazioni)

# Assegna nomi di righe e colonne per la matrice delle correlazioni
rownames(correlation_matrix_PC_RIDGE) <- colnames(best_pc_prediction)
colnames(correlation_matrix_PC_RIDGE) <- paste0(colnames(MSFE_RIDGE_matrix))

# Calcola le correlazioni
for (var_idx in 1:n_variabili) {  # Per ogni variabile
  for (pen_idx in 1:n_penalizzazioni) {  # Per ogni penalizzazione
    # Estrai le previsioni dalla colonna corrispondente di best_pc_prediction
    best_pc_values <- best_pc_prediction[, var_idx]
    
    # Estrai le previsioni dalla colonna corrispondente di all_ridge_prediction
    ridge_values <- all_ridge_prediction[, (var_idx - 1) * n_penalizzazioni + pen_idx]
    
    # Calcola la correlazione e salva nella matrice
    correlation_matrix_PC_RIDGE[var_idx, pen_idx] <- cor(best_pc_values, ridge_values, use = "complete.obs")
  }
}

correlation_matrix_PC_RIDGE <- as.data.frame(correlation_matrix_PC_RIDGE)

# Stampa la matrice delle correlazioni
print(correlation_matrix_PC_RIDGE)

# LASSO

all_LASSO_prediction <- matrix(NA, 
                               nrow = length(seq(from = ind_first, to = length(LASSO))), 
                               ncol = ncol(MSFE_l_matrix) * nrow(MSFE_l_matrix))

# Assegna gli anni come nomi delle righe
rownames(all_LASSO_prediction) <- seq(from = ind_first, to = length(LASSO))  

# Assegna le variabili come nomi delle colonne (qui utilizziamo un vettore per ogni variabile)
col_names <- unlist(lapply(1:nrow(MSFE_l_matrix), function(i) paste0(colnames(MSFE_l_matrix), "_var", i)))
colnames(all_LASSO_prediction) <- col_names

# Loop per ogni anno e variabile
for (j in seq(from = ind_first, to = length(LASSO))) { 
  for (i in 1:nrow(MSFE_l_matrix)) {  # Loop per ogni variabile
    for (jK in seq_along(K)) {  # Loop per ogni penalizzazione jK
      # Estrai la lista di predizioni per la variabile i e l'anno j con la penalizzazione jK
      pred_blist_for_year_var <- LASSO[[j]][[i]][[jK]]
      
      # Posizione della colonna corrispondente alla variabile i
      col_position <- (i - 1) * ncol(MSFE_l_matrix) + jK
      all_LASSO_prediction[j - ind_first + 1, col_position] <- pred_blist_for_year_var
    }
  }
}


# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(all_LASSO_prediction)

all_LASSO_prediction <- as.data.frame(all_LASSO_prediction)


# use best PC prediction to compute the correlation for every variable


# Supponiamo che il numero di variabili sia pari al numero di righe in best_pc_prediction
n_variabili <- ncol(best_pc_prediction)
n_penalizzazioni <- ncol(MSFE_l_matrix)  # Numero di penalizzazioni

# Inizializza la matrice delle correlazioni
correlation_matrix_PC_LASSO <- matrix(NA, nrow = n_variabili, ncol = n_penalizzazioni)

# Assegna nomi di righe e colonne per la matrice delle correlazioni
rownames(correlation_matrix_PC_LASSO) <- colnames(best_pc_prediction)
colnames(correlation_matrix_PC_LASSO) <- paste0(colnames(MSFE_l_matrix))

# Calcola le correlazioni
for (var_idx in 1:n_variabili) {  # Per ogni variabile
  for (pen_idx in 1:n_penalizzazioni) {  # Per ogni penalizzazione
    # Estrai le previsioni dalla colonna corrispondente di best_pc_prediction
    best_pc_values <- best_pc_prediction[, var_idx]
    
    # Estrai le previsioni dalla colonna corrispondente di all_ridge_prediction
    LASSO_values <- all_ridge_prediction[, (var_idx - 1) * n_penalizzazioni + pen_idx]
    
    # Calcola la correlazione e salva nella matrice
    correlation_matrix_PC_LASSO[var_idx, pen_idx] <- cor(best_pc_values, LASSO_values, use = "complete.obs")
  }
}

correlation_matrix_PC_LASSO <- as.data.frame(correlation_matrix_PC_LASSO)

# Stampa la matrice delle correlazioni
print(correlation_matrix_PC_LASSO)
