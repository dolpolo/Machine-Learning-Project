# ---- Set Directory
getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)

# ---- Libraries 
library(dplyr)
library(tidyverse)
library(tseries)
library(ggplot2)
library(writexl)
library(readxl)
library(lubridate)
library(glmnet)

# ---- LOAD DATA ----
# load EA_data untill 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

best_l_model <- readRDS("Results/Best Models/best_l_model.rds")
best_r_model <- readRDS("Results/Best Models/best_r_model.rds")
best_FS_model <- readRDS("Results/Best Models/best_FS_model.rds")

best_l_prediction <- readRDS("Results/Best Models/best_l_prediction.rds")
best_r_prediction <- readRDS("Results/Best Models/best_r_prediction.rds")
best_pc_prediction <- readRDS("Results/Best Models/best_pc_prediction.rds")
best_FS_prediction <- readRDS("Results/Best Models/best_FS_prediction.rds")

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

# ******************************* OUT OF SAMPLE ****************************** #


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


# ==============================================================================
# ===================== COMPARISON IN PREDICTION VISUALIZATION =================
# ==============================================================================

# In this section are presented the out of sample performances comparison between models

add_prefix <- function(df, prefix) {
  df %>%
    rename_with(~ paste0(prefix, .), everything())
}

best_FS_prediction <- add_prefix(best_FS_prediction, "FS_")
best_r_prediction <- add_prefix(best_r_prediction, "r_")
best_l_prediction <- add_prefix(best_l_prediction, "l_")
best_pc_prediction <- add_prefix(best_pc_prediction, "pc_")

prediction_comparison_list <- list(best_r_prediction, best_l_prediction, best_pc_prediction, best_FS_prediction)
prediction_comparison_list <- lapply(prediction_comparison_list, function(df) {
  df %>%
    mutate(Index = seq(start_sample + HH, length(EAdataQ$Time)))
})

prediction_comparison <- reduce(prediction_comparison_list, left_join, by = "Index") %>%
  select(-Index)

getwd()
path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
write_xlsx(prediction_comparison, "Results/Best Models/overall_prediction_comparison.xlsx")

time_subset <- as.Date(EAdataQ$Time[ind_first:length(EAdataQ$Time)])

prediction_comparison_long <- prediction_comparison %>%
  mutate(Time = time_subset) %>%
  mutate(across(where(is.numeric), ~ . / HH)) %>%
  pivot_longer(cols = -Time, names_to = "Variable", values_to = "Value_pred")

# Verifica che `nn` contenga i nomi delle colonne corretti
EAdataQ_long <- EAdataQ %>%
  select(Time, all_of(nn+1)) %>%  # Seleziona la colonna Time e le colonne specificate in nn
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value")


EAdataQ_long_all <- EAdataQ_long %>%
  expand_grid(Prefix = c("pc_", "r_", "l_", "FS_")) %>%
  mutate(Variable = str_c(Prefix, Variable)) %>%
  arrange(Variable, Time) %>%
  select(-Prefix)

output_pred_comp <- left_join(EAdataQ_long_all, prediction_comparison_long, by = c("Time", "Variable")) %>%
  arrange(Time) %>%
  pivot_longer(cols = c(Value, Value_pred), names_to = "Type", values_to = "Value") %>%
  arrange(Variable, Time)

# Create different datasets for every variable to predict

## GDP

output_pred_comp_GDP <- output_pred_comp %>%
  filter(Variable %in% c("l_GDP_EA", "pc_GDP_EA", "r_GDP_EA", "FS_GDP_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "l_") ~ "LASSO",
    str_detect(Variable, "FS_") ~ "FarmSelect",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_GDP, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for GDP in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model Type",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "blue", 
      "RIDGE.Value_pred" = "red", 
      "LASSO.Value_pred" = "green",
      "FarmSelect.Value_pred" = "brown", 
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))


## UNEMPLOYMENT

output_pred_comp_UNETOT <- output_pred_comp %>%
  filter(Variable %in% c("l_UNETOT_EA", "pc_UNETOT_EA", "r_UNETOT_EA", "FS_UNETOT_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "l_") ~ "LASSO",
    str_detect(Variable, "FS_") ~ "FarmSelect",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_UNETOT, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for UNETOT in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model Type",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "blue", 
      "RIDGE.Value_pred" = "red", 
      "LASSO.Value_pred" = "green",
      "FarmSelect.Value_pred" = "brown", 
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))

## PRICES

output_pred_comp_PP <- output_pred_comp %>%
  filter(Variable %in% c("l_PPINRG_EA", "pc_PPINRG_EA", "r_PPINRG_EA", "FS_PPINRG_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "l_") ~ "LASSO",
    str_detect(Variable, "FS_") ~ "FarmSelect",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_PP, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for PPINRG in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model Type",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "blue", 
      "RIDGE.Value_pred" = "red", 
      "LASSO.Value_pred" = "green",
      "FarmSelect.Value_pred" = "brown", 
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))
