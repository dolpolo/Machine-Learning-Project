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

# ---- DATA COLLECTION: EA_countries_data and EA_data ----
# AT <- read_xlsx("data/EA-MD-QD/ATdata.xlsx")
# BE <- read_xlsx("data/EA-MD-QD/BEdata.xlsx") // keep for further analysis* //
# DE <- read_xlsx("data/EA-MD-QD/DEdata.xlsx") // keep for further analysis //
# EL <- read_xlsx("data/EA-MD-QD/ELdata.xlsx")
# ES <- read_xlsx("data/EA-MD-QD/ESdata.xlsx") // keep for further analysis //
# FR <- read_xlsx("data/EA-MD-QD/FRdata.xlsx") // keep for further analysis //
# IE <- read_xlsx("data/EA-MD-QD/IEdata.xlsx")
# IT <- read_xlsx("data/EA-MD-QD/ITdata.xlsx") // keep for further analysis //
# NL <- read_xlsx("data/EA-MD-QD/NLdata.xlsx") // keep for further analysis* //
# PT <- read_xlsx("data/EA-MD-QD/PTdata.xlsx") // keep for further analysis* //

# EA <- read_xlsx("data/EA-MD-QD/EAdata.xlsx") // to be filtered using MATLAB code//

# EA dataset filtered in Quarter variabiles
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_LT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2020-01-01"))

# ---- MERGING TO EA_DATA ----
# EA_list <- list(AT, BE, DE, EL, ES, FR, IE, IT, NL, PT)
# EA_countries_data <- reduce(EA_list, left_join, by = "Time")
# write_xlsx(EA_countries_data, "data/EA-MD-QD/EA_data.xlsx")
# rm(EA_list, EA_countries_data, AT, BE, DE, EL, ES, FR, IE, IT, NL, PT)


# ----------------------------- REPLICATION PAKAGES ----------------------------
# 3PRF: https://github.com/IshmaelBelghazi/ThreePass/blob/master/readme.org
# 3PRF: https://github.com/jacobkahn/ThreePassRegressionRPackage/blob/master/3PRF.R

# ----  DESCRIPTION OF THE EA_DATA ----
# QUARTER STATIONRARY VARIABLE AGGREGATES: National Accounts, Labor Market Indicators, 
# Credit Aggregates, Labor Costs, Exchange Rates, Interest Rates, Industrial Production 
# and Turnover, Prices,  Confidence Indicators, Monetary Aggregates, Others

# FILTERS: 
# Frequency: Quarterly, hence months are converted into Quarters
# Specific Transformation = heavy, meaning that we take the second differences 
# Impute outliers and the Covid period = YES, imputing missing values at 
#   beginning of sample/ragged edges and wiping out Real variablese
# Number of factors for imputation q=99, where q is selected According to [BN02]

# Translate MATLAB codes:

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

# LASSO


# ---- STANDARDIZATION OF DATA

# correlation among EA countries
# EAdataQ <- select_if(EAdataQ, is.numeric)

# ---- BAYASIAN SHRINKAGE 

# (i)RIDGE

# Selezione delle variabili da predire
# 1   : GDP
nn <- c(1)

# Parametri
p <- 0  # Numero di predittori ritardati

rr <- c(1, 3, 5, 10, 25, 50, 75)  # Numero di componenti principali
K <- rr  # Numero di predittori da selezionare tramite Lasso
INfit <- seq(0.1, 0.9, by = 0.1)  # Proporzione di fit in-sample da spiegare con Ridge

HH <- c(4)  # Numero di step avanti per il forecast

# Numero di punti temporali per la stima del parametro: schema Rolling
Jwind <- 68

start_y <- 2017
start_m <- 1  # Date di inizio per la valutazione out-of-sample

# Carica i dati dal file Excel di Stock e Watson (2005)
library(readxl)
# data_raw <- read_excel("hof.xls")
matlabDates <- EAdataQ[,1]
dates <- matlabDates

# time <- as.POSIXlt(dates)
time <- dates

DATA <- as.matrix(EAdataQ[,-1])  # Matrice dei dati: tempo in righe, variabili in colonne
series <- colnames(EAdataQ)  # Etichette delle variabili

# transf <- as.numeric(data_raw[1, -1])  # Indici per le trasformazioni da applicare a ciascuna serie

# Reset delle date e del tempo per eliminare i punti iniziali rimossi
# time <- time[14:length(time)]
# dates <- dates[14:length(dates)]

# Dimensione del pannello
TT <- nrow(DATA)
NN <- ncol(DATA)

# Trova quando iniziare l'esercizio out-of-sample simulato
start_sample <- which(format(time, "%Y") == start_y & format(time, "%m") == start_m)
start_sample <- 68

if (Jwind > start_sample) {
  stop("la finestra rolling non può essere più grande del primo campione di valutazione")
}

cat("\nIl programma sta impostando i parametri di penalizzazione...\n\n")

j0 <- start_sample - Jwind + 1

x_temp <- DATA[j0:start_sample, ]  # Dati disponibili all'inizio della valutazione out-of-sample
x <- DATA[j0:start_sample, ]  # Dati disponibili all'inizio della valutazione out-of-sample
# x <- apply(x_temp, 2, function(col) outliers(col))  # Rimozione degli outliers

x[, nn] <- x_temp[, nn]  # Mantiene le variabili selezionate
x[, nn] <- x [, nn]  # Mantiene le variabili selezionate


# Imposta il parametro di penalizzazione RIDGE per spiegare la proporzione INfit di varianza
nu_ridge <- list()
for (jfit in seq_along(INfit)) {
  for (k in seq_along(nn)) {  # Loop sulle serie da predire
    for (h in HH) {  # Loop sugli orizzonti di previsione
      nu_ridge[[h]][[k]][jfit] <- SET_ridge(x[, nn[k]], x, p, INfit[jfit], h)
    }
  }
}

#(ii) LASSO

# function -->   pred <- (XX[nrow(XX), ] %*% ridge_coeff) * sy + my

#(iii) PCR

# function -->   pred <- (XX[nrow(XX), ] %*% ridge_coeff) * sy + my

