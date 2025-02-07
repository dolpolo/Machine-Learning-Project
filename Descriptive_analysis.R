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
path <- "C:/Users/david/Desktop/University/Machine-Learning-Project"
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
      # Real Gross Domestic Product (GDP - 1) 
      # Real Government Final consumption expenditure (GFCE - 4)
      # Real Households consumption expenditure (HFCE - 5)
  
  # Labor Market Indicators
      # Unemployment: Total (UNETOT - 33) 
      # Wages and salaries (WS- 37) 
  
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
      # General Government - Liabilities: Long-Term Loans (GGLB.LLN - 58) 
  
  # Prices: 
      # Producer Price Index: Durable Consumer Goods (PPIDCOG - 94) 
      # Producer Price Index: Non Durable Consumer Goods (PPINDCOG - 95)
      # Producer Price Index: Energy (PPINRG - 97)


##########################
##  Serial Correlation  ##
##########################
# ---- Serial Correlation in stationary data

# ACF: whole autocorrelation including intermediate lags.
# PACF: Partial autocorrelation between a variable and a given lag.

target_variables <- EAdataQ %>%
  select(GDP_EA, WS_EA, PPINRG_EA)

for (var in names(target_variables)) {
  ts_var <- target_variables[[var]]

  if (is.numeric(ts_var)) {
    # Plot ACF
    print(paste("Autocorrelation for:", var))
    acf(ts_var, main = paste("ACF for", var)) 
    
    # Plot PACF
    print(paste("Partial Autocorrelation for:", var))
    pacf(ts_var, main = paste("PACF for", var)) 
  }
}
# COMMENTS: Proof that variables are stationary and that an autoregressive model can be good

###################################
##  Cross-Sectional Correlation  ##
###################################

EAdataQ <- EAdataQ %>%
  select(!Time)

correlation_matrix <- cor(EAdataQ, use = "complete.obs")

# Correlation matrix
print(correlation_matrix)

# Convert correlation matrix in long format for ggplot
melted_corr <- melt(correlation_matrix)

ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() + 
  theme(
    axis.text.x = element_blank(),  
    axis.text.y = element_blank(), 
    axis.ticks = element_blank()   
  ) +
  labs(x = NULL, y = NULL) +
  coord_fixed()


#####################
##  Visualization  ##
#####################

# Non-stationary data

# Ensure Time is in a Date or numeric format that can be used as an index
EAdata$Time <- as.Date(EAdata$Time) 
# Create the zoo object
gdp_zoo <- zoo(EAdata$GDP_EA, order.by = EAdata$Time)

# Create graph as a plot
plot(gdp_zoo, 
     col = "steelblue", 
     lwd = 2, 
     ylab = "GDP", 
     xlab = "Quarters", 
     main = "EA Quarterly Real GDP")

# Create graph with ggplot2
ggplot(EAdata, aes(x = Time, y = GDP_EA)) +
  geom_line(color = "steelblue", size = 1.5) +
  labs(
    y = "GDP",
    x = "Quarters",
    title = "EA Quarterly Real GDP"
  ) +
  theme_minimal()


EAdata <- EAdata %>%
  arrange(Time) %>%
  mutate(
    recession = c(FALSE, diff(GDP_EA) < 0)  # Identify periods with GDP drops
  ) %>%
  # Identify recession periods (3 consecutive months of decrease in GDP)
  mutate(
    recession_period = rollapply(recession, width = 3, FUN = function(x) all(x == TRUE), align = "right", fill = FALSE)) %>%
  mutate(
    recession_id = cumsum(recession_period & !lag(recession_period, default = FALSE))  # Identify recessions separately 
  )

# Visualize data with recession_id
print(EAdata %>% 
        filter(recession_period == TRUE) %>% 
        select(Time, GDP_EA, recession_period, recession_id))

EAdata$recession_period[EAdata$Time == "2020-03-01"] <- TRUE
EAdata$recession_id[EAdata$Time == "2020-03-01"] <- 3


# Create graph with ggplot
ggplot(EAdata, aes(x = Time, y = GDP_EA)) +
  geom_line(color = "steelblue", size = 1.5) +
  geom_ribbon(data = subset(EAdata, recession_period == TRUE), 
              aes(ymin = -Inf, ymax = max(EAdata$GDP_EA), fill = factor(recession_id)), 
              alpha = 0.3) + 
  scale_fill_manual(values = c("grey", "darkgrey", "#4D4D4D"), 
                    name = "Recession Periods",  
                    labels = c("Great Recession", "European debt crisis", "COVID-19")) +  
  labs(
    y = "GDP",
    x = "Year",
    title = "U.S. Quarterly Real GDP with Recession Periods"
  ) +
  theme_minimal() +
  theme(legend.position = "right")  



subset(EAdata, recession_period == TRUE)

EAdata %>%
  filter(recession_period == TRUE) %>%
  select(Time, GDP_EA, recession_period, recession_id)

# =============================================================================

# Non-stationary data

# Ensure that 'Time' in in numeric or date format 
EAdata$Time <- as.Date(EAdata$Time)

# Create zoo objects for every variable 
target_variables <- list(
  GDP = zoo(EAdata$GDP_EA, order.by = EAdata$Time),
  WS = zoo(EAdata$WS_EA, order.by = EAdata$Time),
  PPINRG = zoo(EAdata$PPINRG_EA, order.by = EAdata$Time)
)

# Labels and colors associated to variables 
variable_info <- list(
  GDP = list(label = "GDP (%)", title = "Real GDP", color = "darkgrey"),
  WS = list(label = "Wage & Salaries", title = "Wage & Salaries", color = "darkgrey"),
  PPINRG = list(label = "Price Energy", title = "Producer Price Index: Energy", color = "darkgrey")
)


par(mfrow = c(3, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0), cex.lab = 1.1, cex.main = 1.2)

# Loop to generate dynamic graphs
for (var in names(target_variables)) {
  plot(target_variables[[var]],
       col = variable_info[[var]]$color,
       lwd = 2,  
       ylab = variable_info[[var]]$label,  
       xlab = "Year",  
       main = variable_info[[var]]$title,  
       cex.main = 1.2,  
       cex.lab = 1.1)  
}

#Title 
mtext("Non-Stationary Macroeconomic Aggregates", outer = TRUE, cex = 1.5, font = 2, line = 2)

# Single plotting area
par(mfrow = c(1, 1))

######Stationary data#######


EAdata$Time <- as.Date(EAdata$Time)

target_variables_s <- list(
  GDP = zoo(EAdataQ$GDP_EA, order.by = EAdata$Time),
  WS = zoo(EAdataQ$WS_EA, order.by = EAdata$Time),
  PPINRG = zoo(EAdataQ$PPINRG_EA, order.by = EAdata$Time)
)


variable_info_s <- list(
  GDP = list(label = "GDP (%)", title = "Real GDP", color = "darkgrey"),
  WS = list(label = "Wage & Salaries", title = "Wage & Salaries", color = "darkgrey"),
  PPINRG = list(label = "Price Energy", title = "Producer Price Index: Energy", color = "darkgrey")
)


par(mfrow = c(3, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0), cex.lab = 1.1, cex.main = 1.2)


for (var in names(target_variables_s)) {
  plot(target_variables_s[[var]],
       col = variable_info_s[[var]]$color, 
       lwd = 2,  
       ylab = variable_info_s[[var]]$label,  
       xlab = "Year",  
       main = variable_info_s[[var]]$title,  
       cex.main = 1.2, 
       cex.lab = 1.1)  
}

# Title 
mtext("Stationary Macroeconomic Aggregates", outer = TRUE, cex = 1.5, font = 2, line = 2)

# Single plotting area
par(mfrow = c(1, 1))



