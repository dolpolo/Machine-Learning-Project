getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)

library(readxl)
library(readr)
library(pracma)
library(fbi)
library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(FarmSelect)
library(glmnet)
library(pls)
library(zoo)

# ---- LOAD THE DATA (till Covid-19) ----
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_LT.xlsx") 
EAdataQ$Time <- as.Date(EAdataQ$Time)
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))
  

# ---- Serial Correlation

# Controlliamo la lista delle variabili nel dataset
names(EAdataQ)

# Creiamo un grafico ACF e PACF per ogni colonna (variabile)
# Escludiamo la colonna della data, se presente
for (var in names(EAdataQ)) {
  # Seleziona la colonna come serie temporale
  ts_var <- EAdataQ[[var]]
  
  # Controlliamo che la variabile sia numerica (necessaria per calcolare ACF e PACF)
  if (is.numeric(ts_var)) {
    # Plot ACF
    print(paste("Autocorrelation for:", var))
    acf(ts_var, main = paste("ACF for", var)) #L'ACF correlazioni tra un dato valore e i suoi ritardi temporali.
    
    # Plot PACF
    print(paste("Partial Autocorrelation for:", var))
    pacf(ts_var, main = paste("PACF for", var)) # La PACF correlazione diretta tra un valore e il suo lag.
  }
}

Macro_best <- EAdataQ %>%
  select(GDP_EA, TEMP_EA, UNETOT_EA)

for (var in names(Macro_best)) {
  # Seleziona la colonna come serie temporale
  ts_var <- Macro_best[[var]]
  
  # Controlliamo che la variabile sia numerica (necessaria per calcolare ACF e PACF)
  if (is.numeric(ts_var)) {
    # Plot ACF
    print(paste("Autocorrelation for:", var))
    acf(ts_var, main = paste("ACF for", var)) #L'ACF correlazioni tra un dato valore e i suoi ritardi temporali.
    
    # Plot PACF
    print(paste("Partial Autocorrelation for:", var))
    pacf(ts_var, main = paste("PACF for", var)) # La PACF correlazione diretta tra un valore e il suo lag.
  }
}

# ---- Cross Sectional Correlation

EAdataQ <- EAdataQ %>%
  select(!Time)

correlation_matrix <- cor(EAdataQ, use = "complete.obs")

# Visualizziamo la matrice di correlazione
print(correlation_matrix)

# Converti la matrice di correlazione in un formato lungo per ggplot
melted_corr <- melt(correlation_matrix)

# Crea la heatmap
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed()

plot(as.zoo(EAdataQ$GDP_EA),
     col = "steelblue",
     lwd = 2,
     ylab = "GDP",
     xlab = "Quarters",
     main = "U.S. Quarterly Real GDP")

# divide plotting area into 2x2 matrix
par(mfrow = c(2, 2))

# plot the series
plot(as.zoo(EAdataQ$GDP_EA),
     col = "steelblue",
     lwd = 2,
     ylab = "Percent",
     xlab = "Date",
     main = "GDP",
     cex.main = 0.8)

plot(as.zoo(EAdataQ$EMP_EA),
     col = "steelblue",
     lwd = 2,
     ylab = "Dollar per pound",
     xlab = "Date",
     main = "EMPLOYMENT",
     cex.main = 0.8)

plot(as.zoo(EAdataQ$UNETOT_EA),
     col = "steelblue",
     lwd = 2,
     ylab = "Logarithm",
     xlab = "Date",
     main = "UNEMPLOYMENT",
     cex.main = 0.8)

plot(as.zoo(EAdataQ$GFCE_EA),
     col = "steelblue",
     lwd = 2,
     ylab = "Percent per Day",
     xlab = "Date",
     main = "GOV exp",
     cex.main = 0.8)

# FRED-MD dataset are strongly correlated and can be well explained by a few principal components

# ---- PCA
# Assuming M_risk_premia is your dataset
# Select only numeric covariates
numeric_data <- M_risk_premia %>% select_if(is.numeric)

# Standardize the data
standardized_data <- scale(numeric_data)

pca_result <- prcomp(standardized_data, center = TRUE, scale. = TRUE)

# Get the standard deviations of the principal components
scree_values <- pca_result$sdev^2

# Create a scree plot for the top 20 principal components
scree_plot <- data.frame(PC = 1:length(scree_values), Variance = scree_values)

# Filter for top 20 PCs
scree_plot_top20 <- scree_plot %>% filter(PC <= 20)

# Plotting the scree plot
ggplot(scree_plot_top20, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity") +
  geom_line(aes(group = 1), color = "blue") +
  labs(title = "Scree Plot of Top 20 Principal Components",
       x = "Principal Component",
       y = "Variance Explained") +
  theme_minimal()

# ---- COMPARISON AMONG THE THREE MODELS ----

# Parametri di rolling window
window_size <- 120
num_steps <- nrow(M_risk_premia) - window_size

# Matrici per salvare i risultati
predictions_farmselect <- numeric(num_steps)
predictions_lasso <- numeric(num_steps)
predictions_pcr <- numeric(num_steps)

for (i in 1:num_steps) {
  # Seleziona la finestra corrente
  train_data <- M_risk_premia[i:(i + window_size - 1), ]
  test_data <- M_risk_premia[i + window_size, ]
  
  # Variabile dipendente e covariate per la finestra corrente
  y_train <- train_data$bond_risk_premia
  X_train <- as.matrix(train_data[ , -which(names(train_data) == "bond_risk_premia")])
  
  ### Modello FarmSelect
  farmselect_fit <- farm.select(X_train, y_train)
  predictions_farmselect[i] <- predict(farmselect_fit, newdata = test_data)
  
  ### Modello LASSO
  lasso_fit <- cv.glmnet(X_train, y_train, alpha = 1)
  predictions_lasso[i] <- predict(lasso_fit, s = lasso_fit$lambda.min, newx = as.matrix(test_data))
  
  ### Modello PCR
  pcr_fit <- pcr(y_train ~ X_train, ncomp = min(128, ncol(X_train)), validation = "CV")
  predictions_pcr[i] <- predict(pcr_fit, newdata = test_data)
}
