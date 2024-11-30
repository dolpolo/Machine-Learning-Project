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
nn <- c(1,37,97)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 25, 40, 50)  # Number of principal components
K <- rr  # Number of predictors with LASSO
INfit <- seq(0.1, 0.9, by = 0.1)  # In-sample residual variance explained by Ridge
HH <- c(4)  # Steap-ahead prediction
Jwind <- 56  # Rolling window



# ****************************** EVALUATION SAMPLE **************************** #

# Dates of the beginning of the evaluation sample
first_y <- 2015
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

#getwd()
#path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
#setwd(path)
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

# Filter the output_pred_comp dataset for dates from 2015-01-01 onward
output_pred_comp_filtered <- left_join(EAdataQ_long_all, prediction_comparison_long, by = c("Time", "Variable")) %>%
  arrange(Time) %>%
  pivot_longer(cols = c(Value, Value_pred), names_to = "Type", values_to = "Value") %>%
  arrange(Variable, Time) %>%
  filter(Time >= as.Date("2015-01-01"))


# Create different datasets for every variable to predict

## GDP

output_pred_comp_GDP <- output_pred_comp_filtered %>%
  filter(Variable %in% c("l_GDP_EA", "FS_GDP_EA")) %>%
  mutate(Model = case_when(
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
    color = "Model",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.major = element_line(color = "lightgray", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "LASSO.Value_pred" = "forestgreen",
      "FarmSelect.Value_pred" = "darkviolet", 
      "Value" = "black"
    ),
    labels = c(
      "LASSO.Value_pred" = "LASSO",
      "FarmSelect.Value_pred" = "FarmSelect", 
      "Value" = "True Value"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Value" = "solid", 
      "Value_pred" = "dashed"
    ),
    labels = c(
      "Value" = "True Value",
      "Value_pred" = "Forecast"
    )
  )

output_pred_comp_GDP <- output_pred_comp_filtered %>%
  filter(Variable %in% c("r_GDP_EA", "pc_GDP_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "pc_") ~ "PC",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_GDP, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for GDP in the Euro Area",
    x = "Year",
    y = "GDP",
    color = "Model",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.major = element_line(color = "lightgray", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "darkorange",
      "RIDGE.Value_pred" = "blue", 
      "Value" = "black"
    ),
    labels = c(
      "PC.Value_pred" = "PCR",
      "RIDGE.Value_pred" = "Ridge", 
      "Value" = "True Value"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Value" = "solid", 
      "Value_pred" = "dashed"
    ),
    labels = c(
      "Value" = "True Value",
      "Value_pred" = "Forecast"
    )
  )


## WS

output_pred_comp_WS <- output_pred_comp_filtered %>%
  filter(Variable %in% c("pc_WS_EA", "r_WS_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "pc_") ~ "PC",
    str_detect(Variable, "r_") ~ "RIDGE",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_WS, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for WS in the Euro Area",
    x = "Year",
    y = "Wages and salaries",
    color = "Model",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.major = element_line(color = "lightgray", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "darkorange",
      "RIDGE.Value_pred" = "blue", 
      "Value" = "black"
    ),
    labels = c(
      "PC.Value_pred" = "PCR",
      "RIDGE.Value_pred" = "Ridge", 
      "Value" = "True Value"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Value" = "solid", 
      "Value_pred" = "dashed"
    ),
    labels = c(
      "Value" = "True Value",
      "Value_pred" = "Forecast"
    )
  )
output_pred_comp_WS <- output_pred_comp_filtered %>%
  filter(Variable %in% c("l_WS_EA", "FS_WS_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "l_") ~ "LASSO",
    str_detect(Variable, "FS_") ~ "FarmSelect",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_WS, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for WS in the Euro Area",
    x = "Year",
    y = "Wages and salaries",
    color = "Model",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.major = element_line(color = "lightgray", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "LASSO.Value_pred" = "forestgreen",
      "FarmSelect.Value_pred" = "darkviolet",
      "Value" = "black"
    ),
    labels = c(
      "LASSO.Value_pred" = "LASSO",
      "FarmSelect.Value_pred" = "FarmSelect", 
      "Value" = "True Value"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Value" = "solid", 
      "Value_pred" = "dashed"
    ),
    labels = c(
      "Value" = "True Value",
      "Value_pred" = "Forecast"
    )
  )

## PRICES

output_pred_comp_PP <- output_pred_comp_filtered%>%
  filter(Variable %in% c("l_PPINRG_EA", "FS_PPINRG_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "l_") ~ "LASSO",
    str_detect(Variable, "FS_") ~ "FarmSelect",
    TRUE ~ "Unknown"
  ))


ggplot(output_pred_comp_PP, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for PPINRG in the Euro Area",
    x = "Year",
    y = "Energy PPI",
    color = "Model",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.major = element_line(color = "lightgray", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "LASSO.Value_pred" = "forestgreen",
      "FarmSelect.Value_pred" = "darkviolet",
      "Value" = "black"
    ),
    labels = c(
      "LASSO.Value_pred" = "LASSO",
      "FarmSelect.Value_pred" = "FarmSelect", 
      "Value" = "True Value"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Value" = "solid", 
      "Value_pred" = "dashed"
    ),
    labels = c(
      "Value" = "True Value",
      "Value_pred" = "Forecast"
    )
  )

output_pred_comp_PP <- output_pred_comp_filtered%>%
  filter(Variable %in% c("r_PPINRG_EA", "pc_PPINRG_EA")) %>%
  mutate(Model = case_when(
    str_detect(Variable, "r_") ~ "RIDGE",
    str_detect(Variable, "pc_") ~ "PC",
    TRUE ~ "Unknown"
  ))


ggplot(output_pred_comp_PP, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +
  labs(
    title = "Comparison of Models for PPINRG in the Euro Area",
    x = "Year",
    y = "Energy PPI",
    color = "Model",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.major = element_line(color = "lightgray", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
  ) +
  scale_color_manual(
    values = c(
      "PC.Value_pred" = "darkorange",
      "RIDGE.Value_pred" = "blue", 
      "Value" = "black"
    ),
    labels = c(
      "PC.Value_pred" = "PCR",
      "RIDGE.Value_pred" = "Ridge", 
      "Value" = "True Value"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Value" = "solid", 
      "Value_pred" = "dashed"
    ),
    labels = c(
      "Value" = "True Value",
      "Value_pred" = "Forecast"
    )
  )


