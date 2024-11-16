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


# **************************************************************************** #
# ********************************** FarmSelect ****************************** #
# **************************************************************************** #  

# This code aims to replicate the analysis by Christine De Mol, Domenico Giannone and Lucrezia Reichlin
# using the paper Forecasting using a large number of predictors: Is Bayesian shrinkage a valid
# alternative to principal components?" on a different dataset on EA countries

# ---- LOAD DATA ----
# load EA_data untill 2019
EAdataQ <- read_xlsx("data/EA-MD-QD/EAdataQ_HT.xlsx") 
EAdataQ <- EAdataQ %>%
  filter(Time <= as.Date("2019-10-01"))

MSFE_PC_matrix <- readRDS("Results/MSFE/MSFE_PC_matrix.rds")
MSFE_PC_ratio <- readRDS("Results/MSFE/MSFE_PC_ratio.rds")
best_pc_prediction <- readRDS("Results/Best Models/best_pc_prediction.rds")

# ---- CALL FUNCTIOINS ----
# call the Bayesian Shrinkage functions
# lapply(list.files("R/functions/", full.names = TRUE), source)
source("R/functions/FarmSelect_functions.R")


# ---- SET PARAMETERS ----
# Dependendent variables to be predicted
nn <- c(1,37,97)

# Parameters
p <- 0  # Number of lagged predictors
rr <- c(1, 3, 5, 10, 25, 40, 50)  # Number of principal components
K <- rr  # Number of predictors with LASSO
HH <- c(4)  # Steap-ahead prediction
Jwind <- 56  # Rolling window


# ********************************* IN SAMPLE ******************************** #

# Date di inizio della valutazione out-of-sample
start_y <- 2014
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

nu_lasso <- list()

for (jK in seq_along(K)) {
  for (k in seq_along(nn)) {
    for (h in HH) {
      
      # Inizializza `nu_lasso[[paste(h, k, sep = "_")]]` come vettore vuoto se non esiste
      if (is.null(nu_lasso[[paste(h, k, sep = "_")]])) {
        nu_lasso[[paste(h, k, sep = "_")]] <- numeric(length(K))  # O usa `rep(NA, length(K))` se vuoi gestire i casi mancanti
      }
      
      # Calcola il valore di `nu` utilizzando `SET_FarmSelect`
      nu_result <- SET_FarmSelect(
        y = x[, nn[k]],
        x = x,
        p = p,
        K = K[jK],
        h = h,
        MSFE_PC_matrix = MSFE_PC_matrix,
        target_index = k  
      )$nu
      
      # Assegna il risultato a `nu_lasso`
      nu_lasso[[paste(h, k, sep = "_")]][jK] <- if (!is.null(nu_result) && length(nu_result) > 0) nu_result else NA
    }
  }
}
nu_lasso
# ================================= out of sample ==============================

# ==============================================================================
# =============================== LASSO PREDICTION ============================= 
# ==============================================================================

# Define lengths for each level
outer_length <- TT
mid_length <- length(nn)
inner_length <- length(K)

# Initialize each list using the function
pred_FS <- initialize_nested_list(outer_length, c(mid_length, inner_length))
FarmSelect <- initialize_nested_list(outer_length, c(mid_length, inner_length))
true_FS <- initialize_nested_list(outer_length, c(mid_length, inner_length))
RW_FS <- initialize_nested_list(outer_length, c(mid_length, inner_length))

# Esegui l'esercizio di previsione fuori campione LASSO
for (j in start_sample:(TT - HH)) {
  
  # Definisci il campione di stima
  j0 <- j - Jwind + 1 # Punto di partenza per la finestra mobile
  
  # Dati disponibili ad ogni punto di valutazione
  x <- X[j0:j, ]  # Dati alla data j
  
  for (k in seq_along(nn)) {
    
    for (jK in seq_along(K)) {
      
      # Costanti di normalizzazione (verifica la necessità di usare const)
      const <- 4
      
      # Calcolo delle previsioni FarmSelect
      for (h in HH) {  # Ciclo sui passi avanti
        
        # Previsione FarmSelect con la funzione FarmSelect_pred
        pred_FS[[j]][[k]][[jK]] <- FarmSelect_pred(
          y = x[, nn[k]], 
          x = x, 
          p = p, 
          nu = nu_lasso[[paste(h, k, sep = "_")]][jK], # Lambda ottimale
          h = h, 
          MSFE_PC_matrix = MSFE_PC_matrix, 
          target_index = k  # Indice variabile target corrente
        )
        
        # Salvare la previsione con normalizzazione
        FarmSelect[[j+h]][[k]][[jK]] <- pred_FS[[j]][[k]][[jK]][["pred"]] * const
        
        # Calcolo del valore reale da prevedere
        temp <- mean(X[(j + 1):(j + h), nn[k]], na.rm = TRUE)
        true_FS[[j+h]][[k]][[jK]] <- temp * const
        
        # Previsione RW con crescita costante
        temp <- RW_pred(x[, nn[k]], h)
        RW_FS[[j+h]][[k]][[jK]] <- temp * const
      }
    }
  }
}



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


# ************************ FarmSelect MSFE AND VISUALIZATION *******************

# Turn the prediction for each variable into a matrix

# ---- MSFE computation

# Number of variables and length of intervals
variable_count <- length(nn)
interval_length <- length(ind_first:length(FarmSelect))

# Initialize output lists
FarmSelect_output <- initialize_output_list(variable_count, interval_length)
true_FS_output <- initialize_output_list(variable_count, interval_length)
RW_output <- initialize_output_list(variable_count, interval_length)

# Process FarmSelect predictions
FarmSelect_output <- process_output(FarmSelect, FarmSelect_output, ind_first, variable_count)

# Process true values
true_FS_output <- process_output(true_FS, true_FS_output, ind_first, variable_count)

# Process Random Walk predictions
RW_output <- process_output(RW_FS, RW_output, ind_first, variable_count)

# Calculate squared differences
diff_FS_output <- calculate_squared_differences(true_FS_output, FarmSelect_output,
                                                variable_count, interval_length)
diff_RW_output <- calculate_squared_differences(true_FS_output, RW_output, variable_count,
                                                interval_length)

# Calculate mean squared forecast error (MSFE)
MSFE_FS <- calculate_msfe(diff_FS_output, variable_count)
MSFE_RW <- calculate_msfe(diff_RW_output, variable_count)

# Display results
print(MSFE_FS)
print(MSFE_RW)

# ---- MFSE Ratio

# Convert MSFE_FS and MSFE_RW lists to matrices
MSFE_FS_matrix <- convert_to_named_matrix(MSFE_FS, target_var, K)
print(MSFE_FS_matrix)

MSFE_RW_matrix <- convert_to_named_matrix(MSFE_RW, target_var, K)
print(MSFE_RW_matrix)

# Calculate and display the ratio of MSFE_FS to MSFE_RW
MSFE_FS_ratio <- MSFE_FS_matrix / MSFE_RW_matrix
print(MSFE_FS_ratio)


# ---- Visualization

# Convertire il dataframe in formato long, includendo un identificatore per le righe
MSFE_FS_matrix_long <- MSFE_FS_matrix %>%
  rownames_to_column(var = "Variable") %>%
  pivot_longer(cols = -Variable, names_to = "Penalization", values_to = "Value")

MSFE_FS_matrix_long <- MSFE_FS_matrix_long %>%
  mutate(Penalization = factor(Penalization, levels = unique(Penalization)))

ggplot(MSFE_FS_matrix_long, aes(x = Penalization, y = Value, color = Variable, group = Variable)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  labs(
    title = "MSFE LASSO FarmSelect",
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


# ==============================================================================
# =================================== BEST MODEL ===============================
# ==============================================================================

# Inizializza una lista vuota per salvare i risultati
best_FS_model <- data.frame(Variable = rownames(MSFE_FS_matrix),
                            Best_MSFE = numeric(nrow(MSFE_FS_matrix)),
                            Best_Penalization = character(nrow(MSFE_FS_matrix)),
                            nu_lasso_FS = numeric(nrow(MSFE_FS_matrix)),
                            stringsAsFactors = FALSE)

# Crea una mappatura tra i nomi delle penalizzazioni e gli indici numerici
penalization_map <- c("1" = 1, "3" = 2, "5" = 3, "10" = 4, 
                      "15" = 5, "40" = 6, "50" = 7)

# Loop per ogni riga della matrice MSFE
for (i in 1:nrow(MSFE_FS_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i
  min_index <- which.min(MSFE_FS_matrix[i, ])
  
  # Assegna il valore minimo e il nome della colonna nella lista dei risultati
  best_FS_model$Best_MSFE[i] <- MSFE_FS_matrix[i, min_index]
  best_FS_model$Best_Penalization[i] <- colnames(MSFE_FS_matrix)[min_index]
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- best_FS_model$Best_Penalization[i]
  
  # Controlla se la penalizzazione è presente nel mapping
  if (best_penalization %in% names(penalization_map)) {
    
    # Ottieni l'indice corrispondente alla penalizzazione
    penalization_index <- penalization_map[best_penalization]
    
    # Seleziona la predizione corrispondente dal nu_lasso
    best_FS_model$nu_lasso_FS[i] <- nu_lasso[[i]][[penalization_index]]
    
  } else {
    # Gestisci il caso in cui la penalizzazione non viene trovata
    print(paste("Penalization", best_penalization, "not found for variable", rownames(MSFE_FS_matrix)[i]))
    best_FS_model$nu_lasso_FS[i] <- NA  # Assegna NA se la penalizzazione non è trovata
  }
}

# Visualizza i risultati
print(best_FS_model)

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
best_FS_prediction <- matrix(NA, nrow = length(seq(from = ind_first, to = length(FarmSelect))), ncol = nrow(MSFE_FS_matrix))
rownames(best_FS_prediction) <- seq(from = ind_first, to = length(FarmSelect))  # Assegna gli anni come nomi delle righe
colnames(best_FS_prediction) <- rownames(MSFE_FS_matrix)  # Assegna le variabili come nomi delle colonne


# Loop per ogni variabile (riga di MSFE_FS_matrix)
for (i in 1:nrow(MSFE_FS_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i (la penalizzazione ottimale)
  min_index <- which.min(MSFE_FS_matrix[i, ])
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- colnames(MSFE_FS_matrix)[min_index]
  
  # Ottieni l'indice della penalizzazione corrispondente
  penalization_index <- penalization_map[best_penalization]
  
  # Per ogni anno (da ind_first a FarmSelect), seleziona la predizione corrispondente alla penalizzazione ottimale
  for (j in ind_first:length(FarmSelect)) {  # Itera sugli anni
    # Estrai la lista di predizioni per la variabile i e l'anno j da FarmSelect
    pred_list_for_year_var <- FarmSelect[[j]][[i]]
    
    # Controlla se penalization_index è dentro i limiti della lista di predizioni
    if (penalization_index <= length(pred_list_for_year_var)) {
      # Salva la predizione corrispondente al miglior parametro di penalizzazione
      best_FS_prediction[j - ind_first + 1, i] <- pred_list_for_year_var[[penalization_index]]
    } else {
      # Se l'indice è fuori limite, assegna NA
      best_FS_prediction[j - ind_first + 1, i] <- NA
      print(paste("Attenzione: Penalization", best_penalization, "è fuori limite per la variabile", rownames(MSFE_FS_matrix)[i], "e l'anno", j))
    }
  }
}

# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(best_FS_prediction)

best_FS_prediction <- as.data.frame(best_FS_prediction)



path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
setwd(path)
saveRDS(best_FS_model, file = "Results/Best Models/best_FS_model.rds")
saveRDS(best_FS_prediction, file = "Results/Best Models/best_FS_prediction.rds")


# ==============================================================================
# ====== VARIABLE FREQUENCY LASSO MODEL SELECTION (more consistent) ============
# ==============================================================================

# Inizializza una matrice vuota per le migliori predizioni per ogni anno
variable_selction <- matrix(NA, nrow = length(seq(from = start_sample, to = length(FarmSelect) - HH)), ncol = nrow(MSFE_FS_matrix))
rownames(variable_selction) <- seq(from = start_sample, to = length(FarmSelect) - HH)  # Assegna gli anni come nomi delle righe
colnames(variable_selction) <- rownames(MSFE_FS_matrix)  # Assegna le variabili come nomi delle colonne

# Loop per ogni variabile (riga di MSFE_FS_matrix)
for (i in 1:nrow(MSFE_FS_matrix)) {
  
  # Trova l'indice della colonna con il valore minimo nella riga i (la penalizzazione ottimale)
  min_index <- which.min(MSFE_FS_matrix[i, ])
  
  # Ottieni il nome della penalizzazione migliore per questa variabile
  best_penalization <- colnames(MSFE_FS_matrix)[min_index]
  
  # Ottieni l'indice della penalizzazione corrispondente
  penalization_index <- penalization_map[best_penalization]
  
  # Per ogni anno (da start_sample a FarmSelect), seleziona la predizione corrispondente alla penalizzazione ottimale
  for (j in start_sample:(length(FarmSelect) - HH)) {  # Itera sugli anni
    # Estrai la lista di predizioni per la variabile i e l'anno j da FarmSelect
    pred_list_for_year_var <- pred_FS[[j]][[i]]
    
    # Verifica che la lista di predizioni per l'anno e la variabile sia valida
    if (is.list(pred_list_for_year_var) && length(pred_list_for_year_var) >= penalization_index) {
      # Seleziona il modello corrispondente alla penalizzazione ottimale
      selected_model <- pred_list_for_year_var[[penalization_index]]
      
      # Controlla se 'model_selection' è una lista (potrebbe contenere più modelli)
      if ("model_selection" %in% names(selected_model)) {
        model_selection_values <- selected_model[["model_selection"]]
        
        # Se 'model_selection' è una lista o vettore, puoi decidere come selezionare i valori
        if (length(model_selection_values) > 1) {
          # Ad esempio, prendi tutti i valori di 'model_selection' e assegnali
          # Potresti anche voler fare un'aggregazione, come la media, se vuoi un singolo valore
          variable_selction[j - start_sample + 1, i] <- paste(model_selection_values, collapse = ", ")
        } else {
          # Se c'è solo un valore, assegna quel valore
          variable_selction[j - start_sample + 1, i] <- model_selection_values
        }
      } else {
        # Se 'model_selection' non è presente nel modello, assegna NA
        variable_selction[j - start_sample + 1, i] <- NA
        print(paste("Attenzione: 'model_selection' non trovato per la variabile", rownames(MSFE_FS_matrix)[i], "e l'anno", j))
      }
    } else {
      # Se l'indice è fuori limite o la lista non è valida, assegna NA
      variable_selction[j - start_sample + 1, i] <- NA
      print(paste("Attenzione: Penalization", best_penalization, "è fuori limite per la variabile", rownames(MSFE_FS_matrix)[i], "e l'anno", j))
    }
  }
}


# Visualizza la matrice delle migliori predizioni per gli anni selezionati
print(variable_selction)

# Trasforma in data frame per manipolazione
variable_selection_df <- as.data.frame(variable_selction)

# Rendi i dati "long" (una riga per ogni variabile selezionata)
freq_table <- variable_selection_df %>%
  pivot_longer(cols = everything(), names_to = "Target", values_to = "SelectedVariables") %>%
  # Dividi le variabili multiple in singole righe
  separate_rows(SelectedVariables, sep = ",") %>%
  # Conta la frequenza per variabile selezionata
  count(Target, SelectedVariables, name = "Frequency") %>%
  # Ordina i risultati per Target e frequenza
  arrange(Target, desc(Frequency))

# Visualizza la tabella risultante
print(freq_table)

# Filtra le 5 variabili più selezionate per ciascuna variabile target
top_5_per_target <- freq_table %>%
  group_by(Target) %>%
  slice_max(Frequency, n = 5) %>%  # Prendi le 5 con frequenza più alta
  ungroup()

# Trasforma in lista di matrici (una per ogni variabile target)
top_5_matrices <- top_5_per_target %>%
  group_split(Target) %>%
  setNames(unique(top_5_per_target$Target)) %>%
  lapply(function(df) {
    matrix(data = c(df$SelectedVariables, df$Frequency),
           ncol = 2,
           dimnames = list(NULL, c("Variable", "Frequency")))
  })

# Visualizza le matrici per ciascun target
top_5_matrices$GDP  # Esempio: matrice per il target GDP
# the model selection is consistent. Every time we ask to select the variable according to the best penalization it extracts the same variables
# in gdp it is not the case probabluy due to the fact it is high correlated?

# Converti la matrice in formato long
variable_selction_long <- as.data.frame(variable_selction) %>%
  mutate(Year = rownames(variable_selction)) %>%
  gather(key = "Variable", value = "SelectedModel", -Year)

# Conta la frequenza delle selezioni per ogni variabile target
freq_table <- variable_selction_long %>%
  group_by(Variable, SelectedModel) %>%
  summarise(Frequency = n(), .groups = 'drop')

ggplot(freq_table, aes(x = Variable, y = Frequency, fill = SelectedModel)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Frequence variable selection for each target",
       x = "Target Variable",
       y = "Frequency",
       fill = "Model Selection") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Ruota le etichette dell'asse x
  guides(fill = "none")  # Rimuove la legenda



# ==============================================================================
# ========================== PREDICTION VISUALIZATION ========================== 
# ==============================================================================

# Verifica che `nn` contenga i nomi delle colonne corretti
EAdataQ_long <- EAdataQ %>%
  select(Time, all_of(nn+1)) %>%  # Seleziona la colonna Time e le colonne specificate in nn
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value")

# Converti la variabile Time in formato Date per rimuovere l'informazione UTC
time_subset <- as.Date(EAdataQ$Time[ind_first:length(FarmSelect)])

best_FS_prediction_long <- best_FS_prediction %>%
  mutate(Time = time_subset)%>%  # Aggiungi la colonna Time
  mutate(across(where(is.numeric), ~ . / 4)) %>% 
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value_pred")

output_pred <- left_join(EAdataQ_long, best_FS_prediction_long, by = c("Time", "Variable"))

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
       subtitle = "Actual and Predicted Values",
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

best_FS_prediction <- best_FS_prediction %>%
  rename_with(~ paste0("FS_", .), everything())

prediction_comparison_list <- list(best_FS_prediction)

# Applica la sequenza come nuova colonna "Index" per ciascun data frame nella lista
prediction_comparison_list <- lapply(prediction_comparison_list, function(df) {
  df %>%
    mutate(Index = (start_sample+HH):length(FarmSelect)) 
})

# Ora puoi effettuare il merge usando "Index" come chiave
prediction_comparison <- reduce(prediction_comparison_list, left_join, by = "Index")
prediction_comparison <- prediction_comparison%>%
  select(-Index)

write_xlsx(prediction_comparison, "Results/Best Models/prediction_comparison_FS.xlsx")

prediction_comparison_long <- prediction_comparison %>%
  mutate(Time = time_subset)%>%  # Aggiungi la colonna Time
  mutate(across(where(is.numeric), ~ . / HH)) %>% 
  pivot_longer(cols = -Time,        # Tutte le colonne eccetto `Time` vanno trasformate
               names_to = "Variable", 
               values_to = "Value_pred")


# Espandiamo il dataset creando una riga per ogni combinazione di variabile e prefisso
EAdataQ_long_all <- EAdataQ_long %>%
  # Creiamo una griglia di combinazioni per ogni variabile con i prefissi
  expand_grid(Prefix = c("FS_")) %>%
  # Ripetiamo per ogni variabile la sequenza di prefissi
  mutate(Variable = str_c(Prefix, rep(EAdataQ_long$Variable, each = 1))) %>%
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
  filter(Variable %in% c("FS_GDP_EA"))

# Add a column to distinguish models in the predictions
output_pred_comp_GDP_long <- output_pred_comp_GDP %>%
  mutate(Model = case_when(
    str_detect(Variable, "FS_") ~ "FarmSelect",
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
      "FarmSelect.Value_pred" = "brown",
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))

## UNEMPLOYMENT

# Make sure the 'output_pred_comp' dataset is already filtered and in long format
output_pred_comp_WS <- output_pred_comp %>%
  filter(Variable %in% c("FS_WS_EA"))

# Add a column to distinguish models in the predictions
output_pred_comp_WS_long <- output_pred_comp_WS %>%
  mutate(Model = case_when(
    str_detect(Variable, "FS_") ~ "FarmSelect",
    TRUE ~ "Unknown"
  ))

ggplot(output_pred_comp_WS_long, aes(x = Time, y = Value, color = interaction(Model, Type), linetype = Type)) +
  geom_line(size = 0.6) +  # Increase line thickness
  labs(
    title = "Comparison of Models for WS in the Euro Area",
    x = "Year",
    y = "Wage and Salaries",
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
      "FarmSelect.Value_pred" = "brown",
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))


## PRICES

# Make sure the 'output_pred_comp' dataset is already filtered and in long format
output_pred_comp_PP <- output_pred_comp %>%
  filter(Variable %in% c("FS_PPINRG_EA"))

# Add a column to distinguish models in the predictions
output_pred_comp_PP_long <- output_pred_comp_PP %>%
  mutate(Model = case_when(
    str_detect(Variable, "FS_") ~ "FarmSelect",
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
      "FarmSelect.Value_pred" = "brown",
      "Value" = "black"
    )
  ) +
  scale_linetype_manual(values = c("Value" = "solid", "Value_pred" = "dashed"))

