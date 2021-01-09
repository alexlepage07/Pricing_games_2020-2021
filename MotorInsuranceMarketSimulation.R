global_imports <- function() {
   require("caret")
   require("rpart")
   require("tidyverse")
   require("gee")
   require("pscl")
   require("tweedie")
   require("rlang")
   require("olsrr")
}
global_imports()


# Loading the data =============================================================
rm(list = ls())

setwd("C:/Users/alexl/Google Drive/Insurance_Market_Simulation")
train_data = read.csv("training.csv")
Xdata = within(train_data, rm('claim_amount'))
ydata = train_data['claim_amount']

glimpse(train_data)
summary(train_data)


# Define your data preprocessing ===============================================
preprocess_X_data <- function (x_raw){
   # Data preprocessing function: given X_raw, clean the data for training or 
   # prediction.
   
   # Parameters
   # ----------
   # X_raw : Dataframe, with the columns described in the data dictionary.
   # 	Each row is a different contract. This data has not been processed.
   
   # Returns
   # -------
   # A cleaned / preprocessed version of the dataset
   
   # YOUR CODE HERE ------------------------------------------------------
   
   Compress_vh_make_model <- function(x_raw, years=c(1)) {
      #' Keeps only the vehicul models the most relevants
      #' for regression.
      freq_tbl <- x_raw %>%
         select(vh_make_model, year) %>%
         filter(year %in% years) %>%
         table() %>% as.data.frame()
      
      n_labels <- nrow(freq_tbl)
      
      labels_to_concat <- freq_tbl[which(freq_tbl[, 'Freq'] < 30), 1]
      
      x_raw[x_raw[,"vh_make_model"] %in% labels_to_concat, "vh_make_model"] <- 
         'others'

      print(paste("Number of labels concatened:", 
            length(labels_to_concat)), quote = F)
      print(paste("Number of remaining labels:", 
            length(table(x_raw[,'vh_make_model']))), quote = F)
      
      return(x_raw)
   }
   
   normalize <- function(x_raw, variables, years=c(1)) {
      #' Normalises the values of a variable
      
      means_ <- list() ; sd_ <- list()
      
      x_train <- x_raw %>% select(all_of(c(variables, "year"))) %>%
         filter(year %in% years)
      
      for (var in variables) {
         means_[var] <- x_train[,var] %>% mean(na.rm=T)
         sd_[var] <- x_train[,var] %>% sd(na.rm=T)
         
         x_raw[,var] <- 
            ((x_raw[,var]) - as.numeric(means_[var])) / as.numeric(sd_[var])
      }
      
      return(x_raw)
   }
   
   treat_NA <- function(x_raw) {
      return(x_raw %>% mutate_all(~replace(., is.na(.), 0)))
   }
   
   
   x_raw <- Compress_vh_make_model(x_raw)
   x_raw <- x_raw %>% mutate(
      pop_density = population / town_surface_area,
      young_second_drv = factor(
         1 * (drv_drv2 == 'Yes') *
             (drv_age2 < 25 | drv_age_lic2 < 5)),
      pol_no_claims_discount = 1 - pol_no_claims_discount,
      vh_value = log(vh_value),
      drv_drv2 = NULL,
      drv_sex2 = NULL
   )
   
   cols <- x_raw %>% select(
      where(is.double),
      -c(year, pol_no_claims_discount, vh_value)
      ) %>% colnames()
   
   X_normalized <- normalize(x_raw, variables = cols, years = c(1))
   X_normalized <- treat_NA(X_normalized)
   
   # ---------------------------------------------------------------------
   return(X_normalized)
}


# Define the training logic ====================================================
fit_model <- function (x_raw, y_raw){
   #' Model training function: given training data (X_raw, y_raw), train this
   #' pricing model.
   
   # Parameters
   # ----------
   # X_raw : Dataframe, with the columns described in the data dictionary.
   # 	Each row is a different contract. This data has not been processed.
   # y_raw : a array, with the value of the claims, in the same order as contracts in X_raw.
   # 	A one dimensional array, with values either 0 (most entries) or >0.
   
   # Returns
   # -------
   # self: (optional), this instance of the fitted model.
   
   
   # This function trains your models and returns the trained model.
   
   # YOUR CODE HERE ------------------------------------------------------------
   TRAINING_YEARS <-  c(1, 2)
   VALIDATION_YEARS <- c(3)
   MODEL_SELECTION <- TRUE
   EVALUATE_MODEL <- TRUE

   x_raw <- Xdata
   y_raw <- ydata
   
   x_clean  <- preprocess_X_data(x_raw)
   df <- data.frame(y_raw, x_clean)
   df_test <- df %>% filter(year %in% VALIDATION_YEARS)
   df_train <- df %>% filter(year %in% TRAINING_YEARS) %>%
      select(-c(id_policy, year, pol_pay_freq))
   
   
   backward_selection <- function(model, TOL=0.05){
      for (i in 1:1000) {
         to_drop <- drop1(model, test='LRT') %>%
            as_tibble(rownames='variable_names') %>%
            top_n(1, `Pr(>Chi)`)
         
         if (to_drop$`Pr(>Chi)` > TOL) {
            model <- model %>% update(
               formula(paste(".~.-", to_drop$variable_names))
            )
            print(
               paste('-', to_drop$variable_names, "; ",
                     "Pr(>Chi)=", round(to_drop$`Pr(>Chi)`, 2)
               ),
               quote = F
            )
         } else break
      }
      return(model)
   }
   
   interaction_selection <- function(model, TOL=c(0.1, 0.05), AIC_K=2) {
      for (i in 1:1000) {
         to_add <- model %>% add1(.~.^2, test='LRT') %>%
            as_tibble(rownames='variable_names') %>%
            top_n(-1, `Pr(>Chi)`)
         
         if (to_add$`Pr(>Chi)` <= TOL[1]) {
            model2 <- model %>%
               update(formula(paste(".~.+", to_add$variable_names)))
            
            if (model$family$family == "Tweedie"){
               get_AIC <- AICtweedie
            } else get_AIC <- AIC
            
            AIC2 <- get_AIC(model2, k=AIC_K)
            AIC1 <- get_AIC(model, k=AIC_K)
            if (AIC2 < AIC1) {
               model <- model2
            
               print(
                  paste('+', to_add$variable_names, "; ",
                        "Pr(>Chi)=", to_add$`Pr(>Chi)`, "; ",
                        "Diff AIC=", round(AIC1-AIC2, 2)
                  ), quote = F
               )
               model <- backward_selection(model, TOL[2])
               
            } else break
         } else break
      }
      return(model)
   }
   
   
   # Training frequency model (i.e. is a claim will occur) ---------------------
   
   (dim_classes <- df_train %>%
      transmute(claim_amount = 1*(claim_amount > 0)) %>% table())
   #' Since there's a lot more of zeros than positive claims, we will need
   #' to adjust the occurrence model to imbalanced dataset.
   dim_classes <- dim_classes * c(0.8, 1.2)
   
   resample <- function(df, strategy=list('under'=60000, 'over'=15000)) {
      
      random_undersample <- function(df, size){
         df0 <- df %>% filter(claim_amount == 0) %>%
            slice_sample(n=size, replace=FALSE)
         df1 <- df %>% filter(claim_amount > 0)
         return(rbind(df0, df1))
      }
      
      random_oversample <- function(df, size){
         df0 <- df %>% filter(claim_amount == 0) 
         df1 <- df %>% filter(claim_amount > 0) %>%
            slice_sample(n=size, replace=TRUE)
         return(rbind(df0, df1))
      }
      
      df <- random_oversample(df, strategy$over)
      df <- random_undersample(df, strategy$under)
      return(df)
   }
   
   df_train_resampled <- df_train %>%
      resample(strategy=list('under'=dim_classes[1], 'over'=dim_classes[2]))
   
   n_occurrence <- sum(df_train_resampled$claim_amount > 0)
   n_observations <- nrow(df_train_resampled)
   p <- n_occurrence/n_observations
   
   cost_function <- function(y, false_neg_cost=1) {
      sapply(y, function(x) if (x == 0) 1 else false_neg_cost)
   }
   
   wt <- (df_train_resampled %>%
             transmute(wt = cost_function(claim_amount, 1/p)))$wt
   
   
   # Feature selection
   if (MODEL_SELECTION) {
      fake_mod <- lm(rep(1, nrow(df_train_resampled)) ~ .,
                     data = df_train_resampled)
      car::vif(fake_mod) %>% as_tibble() %>% top_n(1, `GVIF^(1/(2*Df))`)
      # There's little signs of multicolinearity, but we can live with it.
      
      complete_model <- glm(
         I(claim_amount > 0) ~ . - vh_make_model,
         weights = wt,
         data = df_train_resampled,
         family = binomial(link = 'logit')
      )
      
      # Features selection
      freq_model_backward <- MASS::stepAIC(
         complete_model,
         direction = "backward",
         data = df_train_resampled,
         k=3
      )
      freq_model_backward$anova
      drop1(freq_model_backward, test="LRT")
      
      # Adding interactions
      freq_model <- freq_model_backward %>%
         interaction_selection(TOL=c(0.1, 0.05), AIC_K=log(n_observations))
      
      } else {
         freq_mod_formula <- formula(
            "I(claim_amount > 0) ~ pol_no_claims_discount + pol_coverage +
            pol_duration + pol_sit_duration + pol_payd + pol_usage +
            drv_sex1 + drv_age1 + drv_age_lic1 + drv_age2 + drv_age_lic2 +
            vh_age + vh_fuel + vh_speed + vh_value + vh_weight + population +
            pop_density + young_second_drv + vh_age:vh_fuel +
            pol_usage:vh_weight + pol_duration:pol_usage + vh_age:vh_speed +
            pol_coverage:vh_age + pol_no_claims_discount:pop_density +
            pol_usage:vh_value"
         )
      
      freq_model <- glm(
         freq_mod_formula,
         data = df_train_resampled,
         family = binomial(link = 'logit')
         )
   }
   
   # Evaluation of the performances for the occurrence detection.
   if (EVALUATE_MODEL){
      occurence_predictions <- predict(
         freq_model,
         newdata = df_test,
         type = "response"
         )
      pROC::roc(
         I(df_test$claim_amount>0),
         occurence_predictions
         )
   }
   
   # Training aggregated severity model (Tweedie model) ------------------------
   df_severity_train <- df_train %>% filter(claim_amount > 0)
   
   profile <- tweedie.profile(
      claim_amount ~ 1,
      data = df_severity_train,
      verbose = T,
      method = "mle",
      xi.vec = seq(2, 2.4, by = 0.1)
   )
   xi = profile$xi.max
   
   if (MODEL_SELECTION) {
      # Features selection
      complete_mod <- glm(claim_amount ~ . -vh_make_model,
                          data = df_severity_train,
                          family = statmod::tweedie(xi, 0))
      
      severity_model_backward <- backward_selection(complete_mod)
      severity_model <- severity_model_backward %>%
         interaction_selection(
            TOL=c(0.1, 0.05),
            AIC_K=log(nrow(df_severity_train))
            )
      
   } else {
      sev_mod_formula <- formula(
         "claim_amount ~ pol_no_claims_discount + pol_coverage + drv_age1 +
         vh_age + pop_density + pol_coverage:drv_age1"
      )
      severity_model <- glm(sev_mod_formula,
                           data = df_severity_train,
                           family = statmod::tweedie(xi, 0))
   }
   
   if (EVALUATE_MODEL){
      df_severity_test <- df_test %>% filter(claim_amount > 0)
      
      phi = summary(severity_model)$dispersion
      mu <- predict.glm(severity_model,
                        newdata = df_severity_test,
                        type = "response")
      y_test <- df_severity_test$claim_amount
      U <- ecdf(y_test)(y_test)
      ks.test(U, ptweedie(y_test, xi=xi, mu=mu, phi = phi))
   }
   # ---------------------------------------------------------------------
   # The result trained_model is something that you will save in the next
   # section defining a list and putting the trained models in there
   trained_model <- list(occurence = freq_model,
                        cost = severity_model)
   return(trained_model)
}

# Saving the model =============================================================
save_model <- function(model, output_path="trained_model.RData"){
   #' Saves this trained model to a file.
   #'
   #' This is used to save the model after training, so that it can be used for
   #' prediction later.
   #'
   #' Do not touch this unless necessary (if you need specific features).
   #' If you do, do not forget to update the load_model method to be compatible.
   #'
   #' Save in `trained_model.RData`.
   
   save(model, file=output_path)
}

load_model <- function(model_path="trained_model.RData"){ 
   # Load a saved trained model from the file `trained_model.RData`.
   
   #    This is called by the server to evaluate your submission on hidden data.
   #    Only modify this *if* you modified save_model.
   
   load(model_path)
   return(model)
}


# Predicting the claims ========================================================
predict_expected_claim <- function(model, x_raw){
   # Model prediction function: predicts the average claim based on the pricing model.
   
   # This functions estimates the expected claim made by a contract (typically, as the product
   # of the probability of having a claim multiplied by the average cost of a claim if it occurs),
   # for each contract in the dataset X_raw.
   
   # This is the function used in the RMSE leaderboard, and hence the output should be as close
   # as possible to the expected cost of a contract.
   
   # Parameters
   # ----------
   # X_raw : Dataframe, with the columns described in the data dictionary.
   # 	Each row is a different contract. This data has not been processed.
   
   # Returns
   # -------
   # avg_claims: a one-dimensional array of the same length as X_raw, with one
   #     average claim per contract (in same order). These average claims must be POSITIVE (>0).
   
   
   # YOUR CODE HERE ------------------------------------------------------
   
   # x_clean = preprocess_X_data(x_raw)  # preprocess your data before fitting
   expected_claims = predict(model, newdata = x_raw)  # tweak this to work with your model
   
   return(expected_claims)  
}

claims <- predict_expected_claim(model, Xdata)


# Pricing contracts ============================================================
predict_premium <- function(model, x_raw){
   # Model prediction function: predicts premiums based on the pricing model.
   
   # This function outputs the prices that will be offered to the contracts in X_raw.
   # premium will typically depend on the average claim predicted in 
   # predict_expected_claim, and will add some pricing strategy on top.
   
   # This is the function used in the average profit leaderboard. Prices output here will
   # be used in competition with other models, so feel free to use a pricing strategy.
   
   # Parameters
   # ----------
   # X_raw : Dataframe, with the columns described in the data dictionary.
   # 	Each row is a different contract. This data has not been processed.
   
   # Returns
   # -------
   # prices: a one-dimensional array of the same length as X_raw, with one
   #     price per contract (in same order). These prices must be POSITIVE (>0).
   
   
   # YOUR CODE HERE ------------------------------------------------------
   
   # x_clean = preprocess_X_data(x_raw)  # preprocess your data before fitting
   
   return(predict_expected_claim(model, x_raw) * 2) # Default: bosst prices by a factor of 2
}

prices <- predict_premium(model, Xdata)
as.matrix(head(prices))

# Profit on training data ======================================================
print(paste('Income:', sum(prices)))
print(paste('Losses:', sum(ydata)))

if (sum(prices) < sum(ydata)) {
   print('Your model loses money on the training data! It does not satisfy market rule 1: Non-negative training profit.')
   print('This model will be disqualified from the weekly profit leaderboard, but can be submitted for educational purposes to the RMSE leaderboard.')
} else {
   print('Your model passes the non-negative training profit test!')
}