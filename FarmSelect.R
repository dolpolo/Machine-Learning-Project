getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)

library(dplyr)
library(tidyverse)
library(tseries)
library(ggplot2)
library(writexl)
library(readxl)
library(lubridate)  # Per la gestione delle date
library(glmnet)


# **************************************************************************** #
# ********************************** FarmSelect ****************************** #
# **************************************************************************** #  

# This code aims to replicate the analysis by Christine De Mol, Domenico Giannone and Lucrezia Reichlin
# using the paper Forecasting using a large number of predictors: Is Bayesian shrinkage a valid
# alternative to principal components?" on a different dataset on EA countries


################################################################################

# ---- LOAD DATA ----
# load EA_data untill 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

# ---- CALL FUNCTIOINS ----
# call the Bayesian Shrinkage functions
# lapply(list.files("R/functions/", full.names = TRUE), source)
source("R/functions/FarmSelect_functions.R")


################################################################################


# ---- SET PARAMETERS ----
# Dependendent variables to be predicted
nn <- c(1,33,97)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 20, 50, 65)  # Number of principal components
K <- rr  # Number of predictors with LASSO
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
# ============================= LASSO PENALIZATION ============================= 
# ==============================================================================

# Impostazione del parametro di penalizzazione LASSO
nu_lasso <- list()
for (jK in seq_along(K)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      nu_lasso[[paste(h, k, sep = "_")]][jK] <- SET_FarmSelect(x[, nn[k]], x, p, K[jK], h)
    }
  }
}
nu_lasso

# ================================= out of sample ===========================



# ==============================================================================
# =============================== LASSO PREDICTION ============================= 
# ==============================================================================

# Inizializza pred_FS con una struttura di liste annidate
pred_FS <- vector("list", length = TT)

for (j in 1:TT) {
  pred_FS[[j]] <- vector("list", length = length(nn))
  
  for (k in 1:length(nn)){
    pred_FS[[j]][[k]] <- vector("list", length = length(K))
  }
}

FarmSelect <- vector("list", length = TT)

for (j in 1:TT) {
  FarmSelect[[j]] <- vector("list", length = length(nn))
  
  for (k in 1:length(nn)){
  FarmSelect[[j]][[k]] <- vector("list", length = length(K))
  }
}


true_FS <- vector("list", length = TT)

for (j in 1:TT) {
  true_FS[[j]] <- vector("list", length = length(nn))
  
  for (k in 1:length(nn)){
    true_FS[[j]][[k]] <- vector("list", length = length(K))
  }
}


RW_FS <- vector("list", length = TT)

for (j in 1:TT) {
  RW_FS[[j]] <- vector("list", length = length(nn))
  
  for (k in 1:length(nn)){
    RW_FS[[j]][[k]] <- vector("list", length = length(K))
  }
}

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
      
      # Calcolo delle previsioni FarmSelect
        for (h in HH) {  # Ciclo su numero di passi avanti
          pred_FS[[j]][[k]][[jK]]<- FarmSelect_pred(x[, nn[k]], x, p, nu_lasso[[k]][[jK]], h)
          FarmSelect[[j+h]][[k]][[jK]] <- pred_FS[[j]][[k]][[jK]][["pred"]] * const
        #variance da inserire
        
       
        # Calcola il valore vero da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_FS[[j+h]][[k]][[jK]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_FS[[j+h]][[k]][[jK]] <- temp * const
        
      }
    }
  }
}





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


# ************************ FarmSelect MSFE AND VISUALIZATION *******************

# Turn the prediction for each variable into a matrix

###############
## 
###############

# per predizioni 
# Inizializza una lista per memorizzare le matrici di output per ogni variabile dipendente
FarmSelect_output <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Calcola la lunghezza dell'intervallo degli anni
  pred_quarters <- length(ind_first:length(FarmSelect))
  
  # Inizializza la lista di output per la variabile dipendente k
  FarmSelect_output[[k]] <- vector("list", length = pred_quarters)
}

# Ciclo per ogni anno in FarmSelect, a partire da 'ind_first'
for (j in ind_first:length(FarmSelect)) {
  
  # Ciclo su tutte le variabili dipendenti (k)
  for (k in 1:length(nn)) {
    
    # Prendi la lista delle predizioni per l'anno corrente e la variabile dipendente k
    quarter_prediction <- FarmSelect[[j]][[k]]
    
    # Unisci tutte le predizioni in una matrice (aggiungi le predizioni come righe)
    quarter_pred_metrix <- do.call(rbind, lapply(quarter_prediction, unlist))
    
    # Memorizza la matrice per l'anno e la variabile dipendente k
    # Rinomina la lista con l'anno e la variabile dipendente (usando paste per concatenare)
    FarmSelect_output[[k]][[j - ind_first + 1]] <- quarter_pred_metrix
  }
}


# per true values 
# Inizializza una lista per memorizzare le matrici di output per ogni variabile dipendente
true_FS_output <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Calcola la lunghezza dell'intervallo degli anni
  true_quarters <- length(ind_first:length(FarmSelect))
  
  # Inizializza la lista di output per la variabile dipendente k
  true_FS_output[[k]] <- vector("list", length = true_quarters)
}

# Ciclo per ogni anno in FarmSelect, a partire da 'ind_first'
for (j in ind_first:length(true_FS)) {
  
  # Ciclo su tutte le variabili dipendenti (k)
  for (k in 1:length(nn)) {
    
    # Prendi la lista delle predizioni per l'anno corrente e la variabile dipendente k
    quarter_true <- true_FS[[j]][[k]]
    
    # Unisci tutte le predizioni in una matrice (aggiungi le predizioni come righe)
    quarter_true_metrix <- do.call(rbind, lapply(quarter_true, unlist))
    
    # Memorizza la matrice per l'anno e la variabile dipendente k
    # Rinomina la lista con l'anno e la variabile dipendente (usando paste per concatenare)
    true_FS_output[[k]][[j - ind_first + 1]] <- quarter_true_metrix
  }
}

# Supponendo che 'true_FS_output' e 'predicted_FS_output' siano strutturati correttamente
# e contengano liste di vettori di predizioni per ogni anno

# Inizializza una lista per i risultati delle differenze
diff_FS_output <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Calcola la lunghezza dell'intervallo degli anni
  diff_quarters <- length(ind_first:length(FarmSelect))
  
  # Inizializza la lista di output per la variabile dipendente k
  diff_FS_output[[k]] <- vector("list", length = diff_quarters)
  
  # Ciclo su ogni anno (j)
  for (j in 1:diff_quarters) {
    # Prendi i vettori true e predicted per l'anno j e la variabile k
    true_vector <- true_FS_output[[k]][[j]]
    predicted_vector <- FarmSelect_output[[k]][[j]]
    
    # Assicurati che true_vector e predicted_vector abbiano la stessa lunghezza
    if (length(true_vector) == length(predicted_vector)) {
      # Calcola la differenza al quadrato per ogni elemento nei vettori
      diff_matrix <- (true_vector - predicted_vector)^2
    } 
    # Memorizza il risultato
    diff_FS_output[[k]][[j]] <- diff_matrix
  }
}

# Visualizza i risultati
print(diff_FS_output)

# Inizializza una lista per memorizzare la media di ogni colonna per ogni variabile dipendente
MSFE_FS <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Prendi tutte le liste di vettori per la variabile dipendente k
  vectors_k <- diff_FS_output[[k]]
  
  # Determina la lunghezza dei vettori (assumendo che siano tutti della stessa lunghezza per ogni j)
  vector_length <- length(vectors_k[[1]])
  
  # Inizializza un vettore per memorizzare le medie di ogni "colonna"
  mean_vector <- numeric(vector_length)
  
  # Ciclo su ogni posizione del vettore
  for (pos in 1:vector_length) {
    # Estrai gli elementi alla posizione 'pos' da tutti gli anni j
    elements_at_pos <- sapply(vectors_k, function(v) v[pos])
    
    # Calcola la media di questi elementi
    mean_vector[pos] <- mean(elements_at_pos, na.rm = TRUE)
  }
  
  # Memorizza il vettore di medie per la variabile dipendente k
  MSFE_FS[[k]] <- mean_vector
}

# Visualizza il risultato
print(MSFE_FS)


# compiute the MSFE for RW

# Inizializza una lista per memorizzare le matrici di output per ogni variabile dipendente
RW_output <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Calcola la lunghezza dell'intervallo degli anni
  pred_quarters_RW <- length(ind_first:length(RW_FS))
  
  # Inizializza la lista di output per la variabile dipendente k
  RW_output[[k]] <- vector("list", length = pred_quarters_RW)
}

# Ciclo per ogni anno in FarmSelect, a partire da 'ind_first'
for (j in ind_first:length(RW_FS)) {
  
  # Ciclo su tutte le variabili dipendenti (k)
  for (k in 1:length(nn)) {
    
    # Prendi la lista delle predizioni per l'anno corrente e la variabile dipendente k
    quarter_prediction <- RW_FS[[j]][[k]]
    
    # Unisci tutte le predizioni in una matrice (aggiungi le predizioni come righe)
    quarter_pred_metrix_RW <- do.call(rbind, lapply(quarter_prediction, unlist))
    
    # Memorizza la matrice per l'anno e la variabile dipendente k
    # Rinomina la lista con l'anno e la variabile dipendente (usando paste per concatenare)
    RW_output[[k]][[j - ind_first + 1]] <- quarter_pred_metrix_RW
  }
}


# Inizializza una lista per i risultati delle differenze
diff_RW_output <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Calcola la lunghezza dell'intervallo degli anni
  diff_quarters <- length(ind_first:length(FarmSelect))
  
  # Inizializza la lista di output per la variabile dipendente k
  diff_RW_output[[k]] <- vector("list", length = diff_quarters)
  
  # Ciclo su ogni anno (j)
  for (j in 1:diff_quarters) {
    # Prendi i vettori true e predicted per l'anno j e la variabile k
    true_vector <- true_FS_output[[k]][[j]]
    predicted_vector_RW <- RW_output[[k]][[j]]
    
    # Assicurati che true_vector e predicted_vector abbiano la stessa lunghezza
    if (length(true_vector) == length(predicted_vector_RW)) {
      # Calcola la differenza al quadrato per ogni elemento nei vettori
      diff_matrix <- (true_vector - predicted_vector_RW)^2
    } 
    # Memorizza il risultato
    diff_RW_output[[k]][[j]] <- diff_matrix
  }
}

# Visualizza i risultati
print(diff_RW_output)

# Inizializza una lista per memorizzare la media di ogni colonna per ogni variabile dipendente
MSFE_RW <- vector("list", length = length(nn))

# Ciclo su tutte le variabili dipendenti (k)
for (k in 1:length(nn)) {
  # Prendi tutte le liste di vettori per la variabile dipendente k
  vectors_k <- diff_RW_output[[k]]
  
  # Determina la lunghezza dei vettori (assumendo che siano tutti della stessa lunghezza per ogni j)
  vector_length <- length(vectors_k[[1]])
  
  # Inizializza un vettore per memorizzare le medie di ogni "colonna"
  mean_vector <- numeric(vector_length)
  
  # Ciclo su ogni posizione del vettore
  for (pos in 1:vector_length) {
    # Estrai gli elementi alla posizione 'pos' da tutti gli anni j
    elements_at_pos <- sapply(vectors_k, function(v) v[pos])
    
    # Calcola la media di questi elementi
    mean_vector[pos] <- mean(elements_at_pos, na.rm = TRUE)
  }
  
  # Memorizza il vettore di medie per la variabile dipendente k
  MSFE_RW[[k]] <- mean_vector
}

print(MSFE_RW)

#################### VISUALIZZA IL RATIO 
vector_lengths <- sapply(MSFE_FS, length)

if (length(unique(vector_lengths)) == 1) {
  # Se tutti i vettori hanno la stessa lunghezza, combinali in una matrice
  MSFE_FS_matrix <- do.call(rbind, MSFE_FS)
  MSFE_FS_matrix <- as.data.frame(MSFE_FS_matrix)
  # Rename rows and columns
  rownames(MSFE_FS_matrix) <- target_var
  colnames(MSFE_FS_matrix) <- paste0(K)
  # Visualizza la matrice
  print(MSFE_FS_matrix)
}


vector_lengths <- sapply(MSFE_RW, length)

if (length(unique(vector_lengths)) == 1) {
  # Se tutti i vettori hanno la stessa lunghezza, combinali in una matrice
  MSFE_RW_matrix <- do.call(rbind, MSFE_RW)
  MSFE_RW_matrix <- as.data.frame(MSFE_RW_matrix)
  # Rename rows and columns
  rownames(MSFE_RW_matrix) <- target_var
  colnames(MSFE_RW_matrix) <- paste0(K)
  
  # Visualizza la matrice
  print(MSFE_RW_matrix)
}

MSFE_FS_ratio <- MSFE_FS_matrix / MSFE_RW_matrix
print(MSFE_FS_ratio)



############################## MATRICE PER MATRICE #############################
# Ora FarmSelect_output contiene una lista di matrici per ogni variabile dipendente (k)
# Le matrici sono per ogni anno (rinominate come "Year 72", "Year 73", ...) e contengono le predizioni


# Inizializza una matrice vuota per memorizzare la prima riga di ogni lista
target_1 <- NULL

# Ciclo su tutte le matrici in FarmSelect_output
for (q in names(FarmSelect_output)) {
  
  # Estrai la matrice per l'anno corrente
  quarter_pred_metrix <- FarmSelect_output[[q]]
  
  # Prendi solo la prima riga della matrice
  variable_1 <- quarter_pred_metrix[1, , drop = FALSE]  # drop = FALSE mantiene la matrice come tale
  
  # Combina la prima riga con le altre (aggiungi la prima riga sotto le righe precedenti)
  target_1 <- rbind(target_1, variable_1)
}

# Visualizza la matrice finale con tutte le prime righe
print(target_1)
###############################################################################


# Ridge Output Matrix
FarmSelect_output <- do.call(rbind, lapply(FarmSelect[ind_first:length(FarmSelect)], unlist))

# Rename rows and columns
rownames(FarmSelect_output) <- ind_first:ind_last
colnames(FarmSelect_output) <- paste0(K)

# True Output Matrix
true_FS_output <- do.call(rbind, lapply(true_FS[ind_first:length(true_FS)], unlist))

# Rename rows and columns
rownames(true_FS_output) <- ind_first:ind_last
colnames(true_FS_output) <- paste0(K)

# MSFE ridge
MSFE_LASSO_FS <- matrix(colMeans((FarmSelect_output - true_FS_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_LASSO_FS) <- paste(K)
rownames(MSFE_LASSO_FS)<- paste(nn)

# Convertire la matrice in un dataframe
df_MSFE_LASSO_FS <- as.data.frame(MSFE_LASSO_FS)
df_MSFE_LASSO_FS

# Convertire la matrice in un dataframe
df_MSFE_LASSO_FS <- as.data.frame(MSFE_LASSO_FS)

# Convertire il dataframe in formato long
df_MSFE_LASSO_FS_long <- df_MSFE_LASSO_FS %>%
  pivot_longer(cols = everything(), names_to = "Column", values_to = "Value")

df_MSFE_LASSO_FS_long$Column <- as.numeric(as.character(df_MSFE_LASSO_FS_long$Column))
df_MSFE_LASSO_FS_long$Column <- factor(df_MSFE_LASSO_FS_long$Column, levels = K)

# Plot MSFE against In-sample residual variance
ggplot(df_MSFE_LASSO_FS_long, aes(x = Column, y = Value, group = 1)) +
  geom_line() +  # Linea per i valori
  geom_point() +  # Aggiungi punti per i valori
  labs(title = "MSFE LASSO FarmSelect",
       x = "number of predictors with non zero coefficient",
       y = "MSFE") +
  theme_minimal()

# *************************** LASSO MFSE vs. RW MSFE ***************************
# Ridge Output Matrix
RW_FS_output <- do.call(rbind, lapply(RW_FS[ind_first:length(FarmSelect)], unlist))

# Rename rows and columns
rownames(RW_FS_output) <- ind_first:ind_last
colnames(RW_FS_output) <- paste0(K)

# MSFE RW
MSFE_FS_RW <- matrix(colMeans((RW_FS_output - true_FS_output)^2),nrow = 1, byrow = FALSE)

# Rename matrix
colnames(MSFE_FS_RW) <- paste(K)
rownames(MSFE_FS_RW)<- paste(nn)
MSFE_LASSO_FS_ratio <- MSFE_LASSO_FS / MSFE_FS_RW

rownames(MSFE_LASSO_FS_ratio)[rownames(MSFE_LASSO_FS_ratio) == all_of(nn)] <- "MSFE"
FarmSelect_output_MSFE <- rbind(FarmSelect_output, MSFE_LASSO_FS_ratio)


# ==============================================================================
# ====== VARIABLE FREQUENCY LASSO MODEL SELECTIN (more consistent) =============
# ==============================================================================

# Inizializza un vettore vuoto per raccogliere le variabili
variabili_combine <- c()

# Funzione ricorsiva per estrarre le variabili da model_selection
estrai_model_selection <- function(lista) {
  if (is.list(lista)) {
    # Controlla se la lista contiene model_selection
    if ("model_selection" %in% names(lista)) {
      variabili_combine <<- c(variabili_combine, lista$model_selection)
    }
    # Scorri attraverso gli elementi della lista
    for (elemento in lista) {
      estrai_model_selection(elemento)
    }
  }
}

# Scorri attraverso la megalista e applica la funzione
for (sub_lista in pred_FS) {
  estrai_model_selection(sub_lista)
}

# Calcola la frequenza delle variabili
frequenze <- table(variabili_combine)

# Mostra i risultati
print(frequenze)

plot(frequenze)

# Calcola la frequenza delle variabili
frequenze <- table(variabili_combine)

# Trasforma in dataframe
df_frequenze <- as.data.frame(frequenze)

# Rinomina le colonne per maggiore chiarezza
colnames(df_frequenze) <- c("Variabile", "Frequenza")

# Ordina il dataframe in base alla frequenza, dalla più alta alla più bassa
df_frequenze <- df_frequenze[order(-df_frequenze$Frequenza), ]

row.names(df_frequenze) <- 1:nrow(df_frequenze)

df_frequenze <- df_frequenze %>%
  filter(Frequenza >= 24)

# Crea un grafico a barre
ggplot(df_frequenze, aes(x = reorder(Variabile, -Frequenza), y = Frequenza)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Ruota il grafico per una migliore leggibilità
  labs(title = "Frequenza delle Variabili in model_selection",
       x = "Variabile",
       y = "Frequenza") +
  theme_minimal()

# ==============================================================================
# ========================== PREDICTION VISUALIZATION ========================== 
# ==============================================================================

# time serires of gdp

EAdataQ_long <- EAdataQ %>%
  select(Time, all_of(nn+1)) # there is the time column
ggplot(EAdataQ_long, aes(x = Time, y = pull(EAdataQ_long, 2))) +
  geom_line() + 
  geom_vline(xintercept = as.Date("2017-01-01"), color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = "Time series of GDP in Euro Area",
       x = "Year", 
       y = "GDP in Euro Area") +
  theme_minimal() 


FarmSelect_output_MSFE <- as.data.frame(FarmSelect_output_MSFE)

# Trova la colonna in cui il valore di MSFE è il minimo
min_col <- which.min(FarmSelect_output_MSFE["MSFE", ])  # Se 'MSFE' è una riga, troveremo la colonna con il valore minimo
colname_min <- colnames(FarmSelect_output_MSFE)[min_col]  # Nome della colonna corrispondente

# Selezionare tutte le righe tranne la riga 'MSFE' dalla colonna con il valore minimo
opt_infit <- FarmSelect_output_MSFE %>%
  filter(row.names(FarmSelect_output_MSFE) != "MSFE") %>%  # Escludi la riga 'MSFE'
  select(colname_min)  # Seleziona solo la colonna con il valore minimo


# Crea FarmSelect_opt_df con la colonna chiave
FarmSelect_opt_df <- data.frame(Row.names = rownames(FarmSelect_output), opt_infit)

# Assicurati che EAdataQ_long abbia una colonna chiave per fare il join
EAdataQ_long <- EAdataQ_long %>%
  mutate(Row.names = rownames(EAdataQ_long))

# Rinominare la colonna e dividere i valori per 4
# Rinominare la colonna
FarmSelect_opt_df <- FarmSelect_opt_df %>%
  rename(opt_fit = X1)

# Dividere tutte le voci della colonna "opt_infit" per 4
FarmSelect_opt_df$opt_fit <- FarmSelect_opt_df$opt_fit / 4

# Unisci i dataset
FarmSelect_df <- left_join(EAdataQ_long, FarmSelect_opt_df, by = "Row.names")

# Grafico con ggplot
ggplot(FarmSelect_df, aes(x = Time)) +
  geom_line(aes(y = UNETOT_EA), colour = "blue", linewidth = 0.5) +  # Linea completa del GDP_EA
  geom_line(aes(y = opt_fit), colour = "green", linewidth = 1, na.rm = TRUE) +  # Linea di Ridge_opt solo dove disponibile
  geom_vline(xintercept = as.Date("2017-01-01"), color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = "Time series of GDP in Euro Area",
       x = "Year", 
       y = "GDP in Euro Area")

