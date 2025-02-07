# ---- SET DIRECTORY
getwd()
path <- "C:/Users/david/Desktop/University/Machine-Learning-Project"
setwd(path)


# ---- LIBRARIES
library(dplyr)
library(tidyverse)
library(tseries)
library(ggplot2)
library(writexl)
library(readxl)
library(lubridate)
library(glmnet)


# **************************************************************************** #
# ***************************** COVID PREDICTIONS  *************************** #
# **************************************************************************** #

# ---- LOAD DATA ----
# load EA_data until fourth quarter in 2020 (given our model is a 4-step-head after that date
# the lagged dependent affected by Covid would have entered the regressions)
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2020-10-01"))

# ---- CALL FUNCTIOINS ----
# call the Bayesian Shrinkage functions
# lapply(list.files("R/functions/", full.names = TRUE), source)
#source("R/functions/Bayesian_shrinkage_functions.R")

# ---- SET PARAMETERS ----
# Dependendent variables to be predicted
nn <- c(1,37,97)

# Parameters
p <- 0  # Number of lagged predictors
HH <- c(4)  # Steap-ahead prediction
Jwind <- 56  # Rolling window

# Starting date for Covid predictions (4-steps ahead)
start_y <- 2014
start_m <- 01

# Preparing the data
DATA <- as.matrix(EAdataQ[, -1])  # Matrice dei dati: tempo in righe, variabili in colonne
X <- DATA

# Retrieve dimensions of the dataset
TT <- nrow(X)
NN <- ncol(X)

# Find the index to begin Covid predictions (given our rolling window dimension Jwind)
start_covid <- which(year(EAdataQ$Time) == start_y & month(EAdataQ$Time) == start_m)
j0 <- start_covid - Jwind + 1


# ---- LOAD BEST MODELS ----

best_r_model <- readRDS("Results/Best Models/best_r_model.rds")
best_l_model <- readRDS("Results/Best Models/best_l_model.rds")
best_PC_model <- readRDS("Results/Best Models/best_PC_model.rds")
best_FS_model <- readRDS("Results/Best Models/best_FS_model.rds")

# Converting them into lists
column_rl <- 4 
column_PC <- 3

# Ridge
best_nu_r <- setNames(as.list(best_r_model[[column_rl]]), rownames(best_r_model))
print(best_nu_r)

# LASSO
best_nu_l <- setNames(as.list(best_l_model[[column_rl]]), rownames(best_l_model))
print(best_nu_l)

# PC
best_parameter_PC <- setNames(as.list(as.numeric(best_PC_model[[column_PC]])), rownames(best_PC_model))
print(best_parameter_PC)

# FS
best_nu_FS <- setNames(as.list(as.numeric(best_FS_model[[column_rl]])), rownames(best_FS_model))
print(best_nu_FS)


# ---- RETRIEVE PREDICTIONS FOR EACH MODEL -------

# Define length of each level
outer_length <- TT
mid_length <- length(nn)

# Initialize each list using the function
pred_covid_r <- initialize_nested_list(outer_length, c(mid_length))
RIDGE_covid <- initialize_nested_list(outer_length, c(mid_length))
true_covid <- initialize_nested_list(outer_length, c(mid_length))
RW_covid <- initialize_nested_list(outer_length, c(mid_length))
pred_covid_l <- initialize_nested_list(outer_length, c(mid_length))
LASSO_covid <- initialize_nested_list(outer_length, c(mid_length))
pred_covid_pc <- initialize_nested_list(outer_length, c(mid_length))
PC_covid <- initialize_nested_list(outer_length, c(mid_length))
pred_covid_FS <- initialize_nested_list(outer_length, c(mid_length))
FS_covid<- initialize_nested_list(outer_length, c(mid_length))


# RIDGE predictions for best nu
for (j in start_covid:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (k in seq_along(nn)) {
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni LASSO
      for (h in HH) {  # Ciclo su numero di passi avanti
        pred_covid_r[[j]][[k]]<- RIDGE_pred(x[, nn[k]], x, p, best_nu_r[[k]], h)
        RIDGE_covid[[j+h]][[k]] <- pred_covid_r[[j]][[k]][["pred"]] * const
        #variance da inserire
        
        
        # Calcola il valore vero da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]])
        true_covid[[j+h]][[k]] <- temp * const
        
        # Constant Growth rate
        temp <- RW_pred(x[, nn[k]], h)
        RW_covid[[j+h]][[k]] <- temp * const
        
    }
  }
}

# LASSO predictions for best nu
for (j in start_covid:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (k in seq_along(nn)) {
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni LASSO
      for (h in HH) {  # Ciclo su numero di passi avanti
        pred_covid_l[[j]][[k]]<- LASSO_pred(x[, nn[k]], x, p, best_nu_l[[k]], h)
        LASSO_covid[[j+h]][[k]] <- pred_covid_l[[j]][[k]][["pred"]] * const
    }
  }
}

# PCR prediction for best no. of principal components
for (j in start_covid:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # from where the rolling windows starts
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # I dati disponibili ad ogni punto di tempo
  
  for (k in seq_along(nn)) {
      
      # Costanti di normalizzazione (capire l'if-else da loro)
      const <- 4
      
      # Calcolo delle previsioni PC
      for (h in HH) {  # Ciclo su numero di passi avanti
        pred_covid_pc[[j]][[k]]<- PC_pred(x[, nn[k]], x, p, best_parameter_PC[[k]], h)
        PC_covid[[j+h]][[k]] <- pred_covid_pc[[j]][[k]][["pred"]] * const
        
    }
  }
}

# FARM prediction for best lambdas

# Esegui l'esercizio di previsione fuori campione LASSO
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # Punto di partenza per la finestra mobile
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # Dati alla data j
  
  for (k in seq_along(nn)) {
      
      # Costanti di normalizzazione (verifica la necessità di usare const)
      const <- 4
      
      # Calcolo delle previsioni FarmSelect
      for (h in HH) {  # Ciclo sui passi avanti
        
        # Previsione FarmSelect con la funzione FarmSelect_pred
        pred_covid_FS[[j]][[k]]<- FarmSelect_pred(
          y = x[, nn[k]], 
          x = x, 
          p = p, 
          nu = best_nu_FS[[k]],
          h = h
        )
        
        # Salvare la previsione con normalizzazione
        FS_covid[[j+h]][[k]] <- pred_covid_FS[[j]][[k]][["pred"]] * const
    }
  }
}



# ------ DISPLAY COVID PREDICTIONS ------

row_indices <- (start_covid + HH):TT

#RIDGE 
# Convert the filtered list to a matrix
RIDGE_covid <- RIDGE_covid[row_indices]
RIDGE_covid_pred <-as.data.frame(do.call(rbind, lapply(RIDGE_covid, unlist)))

# Set column names using row names from best_r_model and row names using indeces
colnames(RIDGE_covid_pred) <- best_r_model[[1]][1:length(nn)]
rownames(RIDGE_covid_pred) <- row_indices

#LASSO
# Convert the filtered list to a matrix
LASSO_covid <- LASSO_covid[row_indices]
LASSO_covid_pred <- as.data.frame(do.call(rbind, lapply(LASSO_covid, unlist)))


# Set column names using row names from best_r_model and row names using indeces
colnames(LASSO_covid_pred) <- best_r_model[[1]][1:length(nn)]
rownames(LASSO_covid_pred) <- row_indices

#PCR
# Convert the filtered list to a matrix
PC_covid <- PC_covid[row_indices]
PC_covid_pred <- as.data.frame(do.call(rbind, lapply(PC_covid, unlist)))

# Set column names using row names from best_r_model and row names using indeces
colnames(PC_covid_pred) <- best_r_model[[1]][1:length(nn)]
rownames(PC_covid_pred) <- row_indices


#FarmSelect
# Convert the filtered list to a matrix
FS_covid <- FS_covid[row_indices]
FS_covid_pred <- as.data.frame(do.call(rbind, lapply(FS_covid, unlist)))

# Set column names using row names from best_r_model and row names using indeces
colnames(FS_covid_pred) <- best_r_model[[1]][1:length(nn)]
rownames(FS_covid_pred) <- row_indices


# ==============================================================================
# ============================ PREDICTION VISUALIZATION ========================
# ==============================================================================

# Extract time indices for the full and subset time periods
time_subset <- as.Date(EAdataQ$Time[ind_first:TT])
subset_time <- as.POSIXct(time_subset, format = "%Y-%m-%d")

# ------- DISPLAY COVID COUNTERFACTUALS FOR GDP ------

# Extract the true values and the predicted value
true_values <- DATA[row_indices, 1]
pred_values <- RIDGE_covid_pred[, 1]

# Create the plot with a formal design and updated title
plot(subset_time, true_values, type = "l", col = "black", lwd = 2, 
     xlab = "Year", ylab = "GDP", 
     main = "COVID-19 Counterfactual GDP Forecasts",
     cex.main = 1.5, cex.lab = 1.2, cex.axis = 1.1, 
     font.main = 2, font.lab = 2, font.axis = 2, 
     las = 1, xlim = range(subset_time), ylim = range(c(true_values, pred_values / 4)))

# Add Ridge model predictions (aligning to subset time)
lines(subset_time, pred_values / 4, col = "red", lwd = 2, lty = 2)

# Add a legend with a more formal style
legend("topleft", legend = c("True Values", "Ridge Regression Forecasts"),
       col = c("black", "red"), lty = c(1, 2), lwd = c(2, 2),
       bty = "n", cex = 1.1, text.col = "black", box.col = "white")

# Annotate the last four observations with more formal positioning
last_obs <- tail(subset_time, 4)
last_true_values <- tail(true_values, 4)
last_pred_values <- tail(pred_values / 4, 4)

# Loop to add labels for true and predicted GDP values
for (i in 1:4) {
  # Adjust label position for true values
  y_offset_true <- ifelse(i == 2, -0.005, ifelse(i == 3, 0.002, -0.0065))
  text(last_obs[i], last_true_values[i] + y_offset_true, labels = round(last_true_values[i], 3), 
       col = "black", cex = 0.65, adj = c(0.5, 0))
  
  # Adjust label position for predicted values
  y_offset_pred <- ifelse(i %in% c(2, 4), 0.005, -0.0075)
  text(last_obs[i], last_pred_values[i] + y_offset_pred, labels = round(last_pred_values[i], 3), 
       col = "red", cex = 0.65, adj = c(0.5, 0))
}

# Mark the vertical line at 2020-01-01
date_to_mark <- as.POSIXct("2020-01-01 01:00:00", format = "%Y-%m-%d %H:%M:%S", tz = "CET")
index_to_mark <- which(subset_time == date_to_mark)

# Add vertical line and annotation for COVID-19 outbreak
  abline(v = subset_time[index_to_mark], col = "black", lty = 2, lwd = 1)
  annotation_y_position <- min(true_values) + 0.05
  text(subset_time[index_to_mark], annotation_y_position, labels = "COVID-19 Outbreak", 
       col = "black", cex = 0.7, pos = 2, offset = 1)

# Add grid with formal styling
grid(lty = 3, col = "gray", lwd = 0.5)




# ------- DISPLAY COVID COUNTERFACTUALS FOR WAGE AND SALARIES ------

# Extract the true values and the predicted values
true_values <- DATA[row_indices, 37]
pred_values <- PC_covid_pred[, 2]

# Create the plot with a formal design and updated title
plot(subset_time, true_values, type = "l", col = "black", lwd = 2, 
     xlab = "Year", ylab = "Wage and Salaries", 
     main = "COVID-19 Counterfactual Wage and Salaries Forecasts",
     cex.main = 1.5, cex.lab = 1.2, cex.axis = 1.1, 
     font.main = 2, font.lab = 2, font.axis = 2, 
     las = 1, xlim = range(subset_time), ylim = range(c(true_values, pred_values / 4)))

# Add Ridge model predictions (aligning to subset time)
lines(subset_time, pred_values / 4, col = "red", lwd = 2, lty = 2)

# Add a legend with a more formal style
legend("topleft", legend = c("True Values", "PCR Forecasts"),
       col = c("black", "red"), lty = c(1, 2), lwd = c(2, 2),
       bty = "n", cex = 1.1, text.col = "black", box.col = "white")

# Annotate the last four observations with more formal positioning
last_obs <- tail(subset_time, 4)
last_true_values <- tail(true_values, 4)
last_pred_values <- tail(pred_values / 4, 4)

# Loop to add labels for true and predicted GDP values
for (i in 1:4) {
  # Adjust label position for true values
  y_offset_true <- ifelse(i == 2, -0.005, ifelse(i == 3, 0.002, -0.009))
  text(last_obs[i], last_true_values[i] + y_offset_true, labels = round(last_true_values[i], 3), 
       col = "black", cex = 0.65, adj = c(0.5, 0))
  
  # Adjust label position for predicted values
  y_offset_pred <- ifelse(i %in% c(2, 4), 0.005, -0.0075)
  text(last_obs[i], last_pred_values[i] + y_offset_pred, labels = round(last_pred_values[i], 3), 
       col = "red", cex = 0.65, adj = c(0.5, 0))
}

# Mark the vertical line at 2020-01-01
date_to_mark <- as.POSIXct("2020-01-01 01:00:00", format = "%Y-%m-%d %H:%M:%S", tz = "CET")
index_to_mark <- which(subset_time == date_to_mark)

# Add vertical line and annotation for COVID-19 outbreak
abline(v = subset_time[index_to_mark], col = "black", lty = 2, lwd = 1)
annotation_y_position <- min(true_values) + 0.05
text(subset_time[index_to_mark], annotation_y_position, labels = "COVID-19 Outbreak", 
     col = "black", cex = 0.7, pos = 2, offset = 1)

# Add grid with formal styling
grid(lty = 3, col = "gray", lwd = 0.5)




# ------- DISPLAY COVID COUNTERFACTUALS FOR PRICE of ENERGY ------

# Extract the true values and the predicted values
true_values <- DATA[row_indices, 97]
pred_values <- FS_covid_pred[, 3]

# Create the plot with a formal design and updated title
plot(subset_time, true_values, type = "l", col = "black", lwd = 2, 
     xlab = "Year", ylab = "Energy Producer Price Index", 
     main = "COVID-19 Counterfactual Energy PPI Forecasts",
     cex.main = 1.5, cex.lab = 1.2, cex.axis = 1.1, 
     font.main = 2, font.lab = 2, font.axis = 2, 
     las = 1, xlim = range(subset_time), ylim = range(c(true_values, pred_values / 4)))

# Add Ridge model predictions (aligning to subset time)
lines(subset_time, pred_values / 4, col = "red", lwd = 2, lty = 2)

# Add a legend with a more formal style
legend("topleft", legend = c("True Values", "FarmSelect Forecasts"),
       col = c("black", "red"), lty = c(1, 2), lwd = c(2, 2),
       bty = "n", cex = 1.1, text.col = "black", box.col = "white")

# Annotate the last four observations with more formal positioning
last_obs <- tail(subset_time, 4)
last_true_values <- tail(true_values, 4)
last_pred_values <- tail(pred_values / 4, 4)

# Loop to add labels for true and predicted GDP values
for (i in 1:4) {
  # Adjust label position for true values
  y_offset_true <- ifelse(i == 2, -0.005, ifelse(i == 3, 0.002, -0.006))
  text(last_obs[i], last_true_values[i] + y_offset_true, labels = round(last_true_values[i], 3), 
       col = "black", cex = 0.65, adj = c(0.5, 0))
  
  # Adjust label position for predicted values
  y_offset_pred <- ifelse(i %in% c(2, 4), 0.005, -0.0075)
  text(last_obs[i], last_pred_values[i] + y_offset_pred, labels = round(last_pred_values[i], 3), 
       col = "red", cex = 0.65, adj = c(0.5, 0))
}

# Mark the vertical line at 2020-01-01
date_to_mark <- as.POSIXct("2020-01-01 01:00:00", format = "%Y-%m-%d %H:%M:%S", tz = "CET")
index_to_mark <- which(subset_time == date_to_mark)

# Add vertical line and annotation for COVID-19 outbreak
abline(v = subset_time[index_to_mark], col = "black", lty = 2, lwd = 1)
annotation_y_position <- min(true_values) + 0.05
text(subset_time[index_to_mark], annotation_y_position, labels = "COVID-19 Outbreak", 
     col = "black", cex = 0.7, pos = 3, offset = 13)

# Add grid with formal styling
grid(lty = 3, col = "gray", lwd = 0.5)



