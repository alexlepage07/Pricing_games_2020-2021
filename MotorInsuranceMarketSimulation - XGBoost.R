global_imports <- function() {
   require("caret")
   require("rpart")
   require("tidyverse")
   require("xgboost")
   require("mlr")
   require("stringr")
   require("parallel")
   require("parallelMap")
   require("goftest")
   require("tweedie")
}
global_imports()


# Loading the data =============================================================
rm(list = ls())

setwd("C:/Users/alexl/Google Drive/Pricing_games_2020-2021")
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

      # print(paste("Number of labels concatened:", 
      #       length(labels_to_concat)), quote = F)
      # print(paste("Number of remaining labels:", 
      #       length(table(x_raw[,'vh_make_model']))), quote = F)
      # 
      return(x_raw)
   }
   
   feature_ingeneering <- function(x_raw) {
      x_raw <- Compress_vh_make_model(x_raw)
      x_raw <- x_raw %>% mutate(
         id_policy = NULL,
         across(where(is.character), str_to_lower),
         pop_density = population / town_surface_area,
         vh_speed_drv_age_ratio = vh_speed / drv_age1,
         potential_force_impact = vh_speed * vh_weight,
         pol_pay_freq = NULL
      )
      return(x_raw)
   }
   
   # Add and transform some features
   x_preprocessed <- feature_ingeneering(x_raw)
   
   # Impute missing values
   imputation <- x_preprocessed %>%
      filter(year==1) %>%
      impute(
         classes = list(numeric = imputeMean()),
         dummy.cols = 'vh_value'
      )
   x_preprocessed <- x_preprocessed %>%
         reimpute(imputation$desc)
   
   # One-Hot-Encoding
   x_preprocessed <- model.matrix( ~ . + 0, data = x_preprocessed) %>%
      as.data.frame() %>%
      mutate(vh_value_dummyTRUE = vh_value.dummyTRUE) %>%
      select(-vh_value.dummyTRUE)
      
   
   # ---------------------------------------------------------------------
   return(x_preprocessed)
}


# Define the training logic ====================================================
fit_model <- function (x_raw, y_raw){
   #' Model training function: given training data (X_raw, y_raw), train this
   #' pricing model.
   #'
   #' Parameters
   #' ----------
   #' @X_raw : Dataframe, with the columns described in the data dictionary.
   #' 	Each row is a different contract. This data has not been processed.
   #' @y_raw : An array, with the value of the claims, in the same order as 
   #'    contracts in X_raw. A one dimensional array, with values either 0 
   #'    (most entries) or >0.
   #'
   #' Returns
   #' -------
   #' self: (optional), this instance of the fitted model.
   
   TRAINING_YEARS <-  c(1,2,3,4)
   VALIDATION_YEARS <- c(4)
   GRIDSEARCH <- FALSE
   EVALUATE_MODEL <- TRUE
   
   x_raw <- Xdata
   y_raw <- ydata
   
   x_clean  <- preprocess_X_data(x_raw)
   df <- data.frame(y_raw, x_clean)
   
   df_valid <- df %>% filter(year %in% VALIDATION_YEARS) %>% select(-c(year))
   df_train <- df %>% filter(year %in% TRAINING_YEARS) %>% select(-c(year))
   
   # Training an occurrence detection model with XGB
   train_xgb_occurrence <- 
      function(df_train, df_valid, gridsearch=F, evaluate_perf=T){
         #' Function to train a XGBoost with an objective of type 
         #' binary:logistic for occurrence detection.
         #' 
         #' Parameters
         #' ----------
         #' @df_train dataframe containing training data
         #' @df_valid dataframe containing valitation data
         #' @gridsearch Boolean: Use Gridsearch to optimize hyperparameters ?
         #' @evaluate_perf Boolean: Use df_valid to calculate ROC AUC on 
         #' predictions
         #' 
         #' returns
         #' -------
         #' @model trained model
         #' @auc Calculated ROC AUC on the validation data.
         
         # Rebalancing the classes for the training set
         resample <- function(df, strategy=list('under'=0.7, 'over'=1.2)) {
            
            random_undersample <- function(df, size){
               df0 <- df %>% filter(occ == F) %>%
                  slice_sample(n=size, replace=FALSE)
               df1 <- df %>% filter(occ == T)
               return(rbind(df0, df1))
            }
            
            random_oversample <- function(df, size){
               df0 <- df %>% filter(occ == F) 
               df1 <- df %>% filter(occ == T) %>%
                  slice_sample(n=size, replace=TRUE)
               return(rbind(df0, df1))
            }
               
            dim_classes <- df %>% select(occ) %>% table()
            dim_classes <- dim_classes * c(strategy$under, strategy$over)
            
            df <- random_oversample(df, dim_classes[2])
            df <- random_undersample(df, dim_classes[1])
            return(df)
         }
         
         
         df_train_occ <- df_train %>%
            mutate(occ = factor(claim_amount>0)) %>%
            select(-claim_amount) %>% 
            resample()
            
         # Preparing a validation dataset
         df_valid_occ <- df_valid %>%
            mutate(occ = factor(claim_amount>0)) %>%
            select(-claim_amount)
      
         # Defining task and learner for the mlr optimizer
         train_task <- makeClassifTask(
            data = df_train_occ, target = "occ", positive = "FALSE")
         valid_task <- makeClassifTask(
            data = df_valid_occ, target = "occ", positive = "FALSE")
         
         learner <- makeLearner("classif.xgboost", predict.type = "prob")
         
         if (gridsearch) {
            # Set parallel backend
            parallelStartSocket(cpus = detectCores())
            
            # Model fixed parameters
            learner$par.vals <- list(
               objective = "binary:logistic",
               eval_metric = "auc",
               nrounds = 100
            )
            
            # Set parameter space for gridsearch
            params <- makeParamSet(
               makeDiscreteParam("booster", values = c(
                  'gbtree', 'gblinear', 'dart')),
               makeIntegerParam("max_depth", lower = 3L, upper = 6L),
               makeNumericParam("min_child_weight", lower = 1L, upper = 10L),
               makeNumericParam("subsample", lower = 0.5, upper = 1),
               makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
               makeNumericParam("lambda", lower=1, upper=4),
               makeNumericParam("alpha", lower=0, upper=3),
               makeNumericParam("eta", lower = 0.1, upper = 0.5)
            )
            
            resampling <- makeResampleDesc("CV", stratify = T, iters = 5L)
            control <- makeTuneControlRandom(maxit = 10L, tune.threshold=T)
            
            # Parameter tuning
            mytune <- tuneParams(
               learner = learner,
               task = train_task,
               resampling = resampling,
               par.set = params,
               control = control,
               show.info = T
            )
            parallelStop()
            
            # Set hyperparameters
            learner <- setHyperPars(learner, par.vals = mytune$x)
            
         } else {
            # Pretuned parameters :
            # ---------------------
            #' [Tune] Result: booster=gbtree; max_depth=6; 
            #' min_child_weight=2.79; subsample=0.749; colsample_bytree=0.584; 
            #' lambda=3.93; alpha=0.0535; eta=0.469 : 
            #' @mmce.test.mean=0.1655115
            learner$par.vals <- list(
               objective = "binary:logistic",
               eval_metric = "auc",
               nrounds = 50,
               eta = 0.1,
               booster = 'gbtree',
               max_depth = 3,
               min_child_weight = 2.79,
               subsample = 0.749,
               colsample_bytree = 0.584,
               lambda = 3.93,
               alpha = 2
            )
         }
         
         # Train model
         xgb_model <- mlr::train(learner = learner, task = train_task)
         
         # Evaluation of performance for the occurrence detection.
         if (evaluate_perf){
            # Predictions
            occ_predictions <- predict(xgb_model, valid_task)
            # AUC
            occ_auc <- mlr::performance(occ_predictions, mlr::auc)
            print(occ_auc, quote=F)
            # Confusion matrix
            print(mlr::calculateConfusionMatrix(pred = occ_predictions), quote=F)
            
            return(list("model"=xgb_model, "auc"=occ_auc))
            }
         return(list("model"=xgb_model))
   }
   
   # Training aggregated severity model (XGB: Tweedie model)
   train_xgb_tweedie <- 
      function(df_train, df_valid, gridsearch=F, evaluate_perf=T){
         #' Function to train a XGBoost with an objective of type 
         #' reg:tweedie
         #' 
         #' Parameters
         #' ----------
         #' @df_train dataframe containing training data
         #' @df_valid dataframe containing valitation data
         #' @gridsearch Boolean: Use Gridsearch to optimize hyperparameters ?
         #' @evaluate_perf Boolean: Use df_valid to calculate RMSE on 
         #' predictions
         #' 
         #' returns
         #' -------
         #' @model trained model
         #' @rmse Calculated rmse on the validation data.
         
         df_train_loss <- df_train %>%
            filter(claim_amount>0)
         
         df_valid_loss <- df_valid %>%
            filter(claim_amount>0)
         
         # Defining task and learner for the mlr optimizer
         train_task <- makeRegrTask(
            data = df_train_loss, 
            target = "claim_amount")
         valid_task <- makeRegrTask(
            data = df_valid_loss, 
            target = "claim_amount")
         
         learner <- makeLearner("regr.xgboost", predict.type = "response")
         
         if (gridsearch) {
            # Set parallel backend
            parallelStartSocket(cpus = detectCores())
            
            # Model fixed parameters
            learner$par.vals <- list(
               objective = "reg:tweedie",
               eval_metric = "rmse",
               nrounds = 200,
               gamma = 1e-1
            )
            
            # Set parameter space for gridsearch
            params <- makeParamSet(
               makeNumericParam("tweedie_variance_power", lower=1, upper=2),
               makeDiscreteParam("booster", values = c(
                  'gbtree', 'gblinear', 'dart')),
               makeIntegerParam("max_depth", lower = 3L, upper = 6L),
               makeNumericParam("min_child_weight", lower = 1L, upper = 10L),
               makeNumericParam("subsample", lower = 0.5, upper = 1),
               makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
               makeNumericParam("lambda", lower=1, upper=4),
               makeNumericParam("alpha", lower=0, upper=3),
               makeNumericParam('eta', lower=0.1, upper=0.5)
            )
            
            control <- makeTuneControlRandom(maxit = 10L)
            resampling <- makeResampleDesc("CV", iters = 5L)
            
            # Parameter tuning
            mytune <- tuneParams(
               learner = learner,
               task = train_task,
               resampling = resampling,
               par.set = params,
               control = control,
               show.info = T
            )
            parallelStop()
            
            # Set hyperparameters
            learner <- setHyperPars(learner, par.vals = mytune$x)
            
         } else {
            # Pretuned parameters :
            # ---------------------
            #' [Tune] Result: tweedie_variance_power=1.12; booster=gblinear;
            #' max_depth=4; min_child_weight=7.1; subsample=0.94;
            #' colsample_bytree=0.617; lambda=3.11; alpha=0.295; eta=0.314 :
            #' @mse.test.mean=3348276.3538605
            learner$par.vals <- list(
               objective = "reg:tweedie",
               eval_metric = "rmse",
               nrounds = 200,
               eta = 0.314,
               gamma = 1e-1,
               tweedie_variance_power=1.12,
               booster='gblinear',
               max_depth=4,
               min_child_weight=7.1,
               subsample=0.94,
               colsample_bytree=0.617,
               lambda=3.11,
               alpha=0.295
            )
         }
         
         # Train model
         xgb_model <- mlr::train(learner = learner, task = train_task)
         
         if (evaluate_perf){
            # Evaluation of performance for the occurrence detection.
            claim_predictions <- predict(xgb_model, valid_task)
            # RMSE
            rmse_tweedie <- mlr::performance(claim_predictions, mlr::rmse)
            print(rmse_tweedie)
            
            # Calculate measures for adequacy evaluation
            truth <- claim_predictions$data$truth
            mu <- claim_predictions$data$response
            xi <- xgb_model$learner.model$params$tweedie_variance_power
            phi <- optimize(function(phi) {
               -sum(log(1 + dtweedie(
                  y = truth,
                  xi = xi,
                  mu = mu,
                  phi = phi
               )))
            }, interval = c(0.1, 1000))$minimum
            U <- ecdf(truth)(truth)
            Fx <- function(x) ptweedie(x, xi=xi, mu=mu, phi = phi)
            
            # Kolmogorov-Smirnov test
            ks.test(U, Fx(truth))
            
            # Anderson-Darling test
            goftest::ad.test(truth, null=Fx, estimated=F, nullname = 'Tweedie')
            #' According to both the K-S and A-D test, the model doesn't fit 
            #' well to the dataset. 
            return(list("model"=xgb_model, "rmse"=rmse_tweedie))
         }
         return(list("model"=xgb_model))
   }
   
   
   severity_model <- train_xgb_tweedie(df_train,
                                       df_valid,
                                       gridsearch = GRIDSEARCH,
                                       evaluate_perf = EVALUATE_MODEL)$model
   occ_model <- train_xgb_occurrence(df_train, 
                                     df_valid, 
                                     gridsearch = GRIDSEARCH,
                                     evaluate_perf = EVALUATE_MODEL)$model

   return(list(occurence = occ_model,
               cost = severity_model))
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


save_model(fit_model(Xdata, ydata))
model <- load_model('trained_model.RData')

# Predicting the claims ========================================================
predict_expected_claim <- function(model, x_raw){
   #' Model prediction function: predicts the average claim based on the pricing model.
   #'
   #' This functions estimates the expected claim made by a contract (typically, as the product
   #' of the probability of having a claim multiplied by the average cost of a claim if it occurs),
   #' for each contract in the dataset X_raw.
   #'   
   #' This is the function used in the RMSE leaderboard, and hence the output should be as close
   #' as possible to the expected cost of a contract.
   #'
   #' Parameters
   #' ----------
   #' @X_raw : Dataframe, with the columns described in the data dictionary.
   #' 	Each row is a different contract. This data has not been processed.
   #'
   #' Returns
   #' -------
   #' @avg_claims: a one-dimensional array of the same length as X_raw, with one
   #'     average claim per contract (in same order). These average claims must be POSITIVE (>0).
   
   x_clean = preprocess_X_data(x_raw) %>% select(-year)
   
   expected_occ <- predict(model$occurence, newdata = x_clean)$data$prob.TRUE
   expected_loss <- predict(model$cost, newdata = x_clean)$data$response
   
   expected_claims = expected_occ * expected_loss
   return(round(expected_claims, 2))  
}


expected_claims <- predict_expected_claim(model, Xdata)
sqrt(mean((expected_claims - unlist(ydata))^2))

# Pricing contracts ============================================================
predict_premium <- function(model, x_raw){
   #' Model prediction function: predicts premiums based on the pricing model.
   #'
   #' This function outputs the prices that will be offered to the contracts in X_raw.
   #' premium will typically depend on the average claim predicted in 
   #' predict_expected_claim, and will add some pricing strategy on top.
   #'
   #' This is the function used in the average profit leaderboard. Prices output here will
   #' be used in competition with other models, so feel free to use a pricing strategy.
   #'
   #' Parameters
   #' ----------
   #' @X_raw : Dataframe, with the columns described in the data dictionary.
   #' 	Each row is a different contract. This data has not been processed.
   #'
   #' Returns
   #' -------
   #' @prices: a one-dimensional array of the same length as X_raw, with one
   #'     price per contract (in same order). These prices must be POSITIVE (>0).
   
   expectations <- predict_expected_claim(model, x_raw)
   return(expectations * 1.1)
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