# **************************************************************************** #
# ********************** DESCRIPTIVE TIME SERIES ANALYSIS ******************** #
# **************************************************************************** #  

# This code performs data cleaning and and time series descriptive analysis on EA data.
# The trend of non-stationary data is presented alongside the serial and cross sectional correlation

# NOTE: Data are Stationalized through the Replication code by Barigozzi and Lissona using
# Quartely variables and the quarterly observations of monthly ones and heavie tranformations

#################
##  Directory  ##
#################

getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)


#################
##  Libraries  ##
#################

# Libraries essential for time series descriptive analysis
library(readxl)           # Import Excel files
library(readr)            # Import CSV or text files
library(dplyr)            # Data manipulation
library(tidyr)            # Data tidying and reshaping
library(ggplot2)          # Data visualization
library(zoo)              # Time series data handling
library(reshape2)  # oppure library(data.table)


# Optional but useful for specific tasks in time series analysis
library(forecast)    # Time series analysis and forecasting
library(tseries)     # Statistical tests for time series data
library(xts)         # Extended time series handling, complements zoo


#################
##  Load Data  ##
#################

# Load data till Covid-19

# Non-stationary data
EAdata <- read_xlsx("data/EA-MD-QD/EAdata.xlsx") 
EAdata$Time <- as.Date(EAdata$Time)
EAdata <- EAdata %>% drop_na()
EAdata <- EAdata %>%
  filter(Time <= as.Date("2019-10-01"))

# 2000-01-01 represents the first quarter (Q1) in 2000
# Recall that after the transformation we take first differences: shift in Time

# Stationary data 
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ$Time <- as.Date(EAdataQ$Time)
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))


########################
##  Target Variables  ##
########################

# Most affected by COVID-19

  # National Accounts
      # Real Gross Domestic Product (GDP - 1) !!!
      # Real Government Final consumption expenditure (GFCE - 4)
      # Real Households consumption expenditure (HFCE - 5)
  
  # Labor Market Indicators
      # Unemployment: Total (UNETOT - 33) !!!
      # Wages and salaries (WS- 37) !!!
  
  # Credit Aggregates
      # Non-Financial Corporations - Liabilities - Short-Term Loans (NFCLB.SLN - 51)
      # Households - Liabilities: short-Term Loans (HHLB.SLN - 63)
      # General Government - Liabilities: Short-Term Loans ( GGLB.SLN - 57)
  
  # Industrial Production and Turnover
      # Industrial Production Index: Capital Goods (IPCAG - 79)
      # Industrial Production Index: Consumer Goods (IPCOG- to)

# Less affected by COVID-19

  #  Credit Aggregates
      # Non-Financial Corporations - Liabilities - Long-Term Loans (NFCLB.LLN - 52)
      # Households - Liabilities: Long-Term Loans (HHLB.LLN - 64)
      # General Government - Liabilities: Long-Term Loans (GGLB.LLN - 58) !!!
  
  # Prices: 
      # Producer Price Index: Durable Consumer Goods (PPIDCOG - 94) !!!
      # Producer Price Index: Non Durable Consumer Goods (PPINDCOG - 95)
      # Producer Price Index: Energy (PPINRG - 97)


##########################
##  Serial Correlation  ##
##########################

# It is convenient to analyise the serial and the cross sectional correlation using stationary data
# Non stationary once could be affected by the trend of growth of variables

# ---- Serial Correlation in stationary data

# ACF: whole autocorrelation including intermediate lags.
# PACF: Partial autocorrelation between a variable and a given lag.

target_variables <- EAdataQ %>%
  select(GDP_EA, WS_EA, GGLB.LLN_EA, PPINRG_EA)

for (var in names(target_variables)) {
  # Seleziona la colonna come serie temporale
  ts_var <- target_variables[[var]]
  
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
# COMMENTS: Proof that variables are stationary and that an autoregressive model can be good
# nei MA non c'è correlazione diretta dopo il q-esimo lag, ma solo indiretta quindi l'ACF sarebbe tagliata e la PACF decrescente


###################################
##  Cross-Sectional Correlation  ##
###################################

EAdataQ <- EAdataQ %>%
  select(!Time)

correlation_matrix <- cor(EAdataQ, use = "complete.obs")

# Visualizziamo la matrice di correlazione
print(correlation_matrix)

# Converti la matrice di correlazione in un formato lungo per ggplot
melted_corr <- melt(correlation_matrix)

ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() + 
  theme(
    axis.text.x = element_blank(),  # Rimuove i testi sull'asse x
    axis.text.y = element_blank(),  # Rimuove i testi sull'asse y
    axis.ticks = element_blank()    # Rimuove i tick dagli assi
  ) +
  labs(x = NULL, y = NULL) +  # Rimuove i nomi degli assi
  coord_fixed()


#####################
##  Visualization  ##
#####################

# Non-stationary data

# Ensure Time is in a Date or numeric format that can be used as an index
EAdata$Time <- as.Date(EAdata$Time) 
# Create the zoo object
gdp_zoo <- zoo(EAdata$GDP_EA, order.by = EAdata$Time)

# Creare il grafico come plot
plot(gdp_zoo, 
     col = "steelblue", 
     lwd = 2, 
     ylab = "GDP", 
     xlab = "Quarters", 
     main = "U.S. Quarterly Real GDP")

# Creare il grafico con ggplot2
ggplot(EAdata, aes(x = Time, y = GDP_EA)) +
  geom_line(color = "steelblue", size = 1.5) +
  labs(
    y = "GDP",
    x = "Quarters",
    title = "U.S. Quarterly Real GDP"
  ) +
  theme_minimal()

# ======================= PER qualche RAGIONE non ESCE =========================

EAdata <- EAdata %>%
  arrange(Time) %>%
  mutate(
    recession = c(FALSE, diff(GDP_EA) < 0)  # Identificare i periodi di calo del PIL
  ) %>%
  # Identificare i periodi di recessione (3 mesi consecutivi di calo del PIL)
  mutate(
    recession_period = rollapply(recession, width = 3, FUN = function(x) all(x == TRUE), align = "right", fill = FALSE)) %>%
  # Creare una colonna per identificare ciascuna recessione separatamente
  mutate(
    recession_id = cumsum(recession_period & !lag(recession_period, default = FALSE))  # Identifica le recessioni separatamente
  )

# Visualizza i dati con recession_id
print(EAdata %>% 
        filter(recession_period == TRUE) %>% 
        select(Time, GDP_EA, recession_period, recession_id))

EAdata$recession_period[EAdata$Time == "2020-03-01"] <- TRUE
EAdata$recession_id[EAdata$Time == "2020-03-01"] <- 3


# Creare il grafico con ggplot
ggplot(EAdata, aes(x = Time, y = GDP_EA)) +
  geom_line(color = "steelblue", size = 1.5) +
  # Aggiungere il colore rosso solo ai periodi di recessione
  geom_ribbon(data = subset(EAdata, recession_period == TRUE), 
              aes(ymin = -Inf, ymax = max(EAdata$GDP_EA), fill = factor(recession_id)), 
              alpha = 0.3) + 
  scale_fill_manual(values = c("grey", "darkgrey", "#4D4D4D"), 
                    name = "Recession Periods",  # Titolo della legenda
                    labels = c("Great Recession", "European debt crisis", "COVID-19")) +  # Etichette per i periodi
  labs(
    y = "GDP",
    x = "Year",
    title = "U.S. Quarterly Real GDP with Recession Periods"
  ) +
  theme_minimal() +
  theme(legend.position = "right")  # Posizione della legenda



subset(EAdata, recession_period == TRUE)

EAdata %>%
  filter(recession_period == TRUE) %>%
  select(Time, GDP_EA, recession_period, recession_id)

# =============================================================================

# Non-stationary data

# Assicurarsi che 'Time' sia in formato Date o numerico
EAdata$Time <- as.Date(EAdata$Time)

# Creare oggetti zoo per ciascuna variabile
target_variables <- list(
  GDP = zoo(EAdata$GDP_EA, order.by = EAdata$Time),
  UNETOT = zoo(EAdata$UNETOT_EA, order.by = EAdata$Time),
  GGLB.LLN = zoo(EAdata$GGLB.LLN_EA, order.by = EAdata$Time),
  PPINDCOG = zoo(EAdata$PPINDCOG_EA, order.by = EAdata$Time)
)

# Etichette e colori associati alle variabili
variable_info <- list(
  GDP = list(label = "GDP (%)", title = "Real GDP", color = "steelblue"),
  UNETOT = list(label = "Unemployment Rate (%)", title = "Unemployment", color = "darkred"),
  GGLB.LLN = list(label = "Government Loans (MLN)", title = "Government Long-Term Loans", color = "darkgreen"),
  PPINDCOG = list(label = "Price Durable Goods", title = "Price Index: Durable Consumer Goods", color = "purple")
)

# Impostazione della zona di plotting in una matrice 2x2 con margini esterni aggiustati
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0), cex.lab = 1.1, cex.main = 1.2)

# Ciclo per generare i grafici dinamicamente
for (var in names(target_variables)) {
  plot(target_variables[[var]],
       col = variable_info[[var]]$color, # Colore della linea
       lwd = 2,  # Spessore della linea
       ylab = variable_info[[var]]$label,  # Etichetta dell'asse Y
       xlab = "Year",  # Etichetta dell'asse X
       main = variable_info[[var]]$title,  # Titolo del grafico
       cex.main = 1.2,  # Dimensione del titolo
       cex.lab = 1.1)  # Dimensione delle etichette degli assi
}

# Aggiungere il titolo generale con spazio sufficiente
mtext("Non-Stationary Macroeconomic Aggregates", outer = TRUE, cex = 1.5, font = 2, line = 2)

# Ripristinare la configurazione a una sola area di plotting
par(mfrow = c(1, 1))

# Stationary data

# Assicurarsi che 'Time' sia in formato Date o numerico
EAdata$Time <- as.Date(EAdata$Time)

# Creare oggetti zoo per ciascuna variabile dei dati stazionari
target_variables_s <- list(
  GDP = zoo(EAdataQ$GDP_EA, order.by = EAdata$Time),
  UNETOT = zoo(EAdataQ$UNETOT_EA, order.by = EAdata$Time),
  GGLB.LLN = zoo(EAdataQ$GGLB.LLN_EA, order.by = EAdata$Time),
  PPINDCOG = zoo(EAdataQ$PPINDCOG_EA, order.by = EAdata$Time)
)

# Etichette e colori associati alle variabili stazionarie
variable_info_s <- list(
  GDP = list(label = "GDP (%)", title = "Real GDP", color = "steelblue"),
  UNETOT = list(label = "Unemployment Rate (%)", title = "Unemployment", color = "darkred"),
  GGLB.LLN = list(label = "Government Loans (MLN)", title = "Government Long-Term Loans", color = "darkgreen"),
  PPINDCOG = list(label = "Price Durable Goods", title = "Price Index: Durable Consumer Goods", color = "purple")
)

# Impostazione della zona di plotting in una matrice 2x2 con margini esterni aggiustati
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0), cex.lab = 1.1, cex.main = 1.2)

# Ciclo per generare i grafici dinamicamente
for (var in names(target_variables_s)) {
  plot(target_variables_s[[var]],
       col = variable_info_s[[var]]$color, # Colore della linea
       lwd = 2,  # Spessore della linea
       ylab = variable_info_s[[var]]$label,  # Etichetta dell'asse Y
       xlab = "Year",  # Etichetta dell'asse X
       main = variable_info_s[[var]]$title,  # Titolo del grafico
       cex.main = 1.2,  # Dimensione del titolo
       cex.lab = 1.1)  # Dimensione delle etichette degli assi
}

# Aggiungere il titolo generale con spazio sufficiente
mtext("Stationary Macroeconomic Aggregates", outer = TRUE, cex = 1.5, font = 2, line = 2)

# Ripristinare la configurazione a una sola area di plotting
par(mfrow = c(1, 1))




############################
## SUPPLEMENTARY ANALYSIS ##
############################
# In this section representation for LaTeX are performed

################################
## GAUSSIAN AND LAPLACE PRIOR ##
################################

dlaplace <- function(x, location = 0, scale = 1) {
  return((1/(2*scale)) * exp(-abs(x - location) / scale))
}


# Definisci l'intervallo per le distribuzioni
x <- seq(-10, 10, by = 0.1)

# Calcola le densità per la distribuzione normale (gaussiana)
mu <- 0          # media
sigma <- 1       # deviazione standard
normal_density <- dnorm(x, mean = mu, sd = sigma)

# Calcola le densità per la distribuzione di Laplace
laplace_density <- dlaplace(x, location = mu, scale = 1)

# Crea un dataframe per ggplot
data <- data.frame(x, normal_density, laplace_density)

# Genera il grafico
ggplot(data, aes(x)) +
  geom_line(aes(y = normal_density, color = "Gaussian"), size = 1.2) +
  geom_line(aes(y = laplace_density, color = "Laplace"), size = 1.2) +
  labs(title = "Comparison of Gaussian and Laplace Distributions",
       subtitle = "Illustrating the differences in shape and spread",
       x = "X values",
       y = "Density",
       color = "Distribution") +
  scale_color_manual(values = c("Gaussian" = "blue", "Laplace" = "red")) +
  theme_minimal(base_size = 15) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 14, face = "italic"),
    legend.position = "top",
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_blank()
  )

