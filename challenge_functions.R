library(mvtnorm)
library(hetGP)
library(ggplot2)
library(MASS)
library(tidyverse)

# transforms y
f <- function(x) {
  y <- log(x + 1)
  return(y)
}

# This function back transforms the input argument
fi <- function(y) {
  x <- exp(y) - 1
  return(x)
}

train_model <- function(data, predictors) {
  df_train <- data |> dplyr::select("date", "cases", predictors)

  ### scaling predictors between [0,1]
  scaler <- caret::preProcess(df_train |> dplyr::select(predictors), method = "range")
  df_train2 <- df_train
  df_train2[predictors] <- predict(scaler, df_train |> dplyr::select(predictors))

  X <- df_train2 |>
    select(predictors) |>
    as.matrix()
  y_obs <- df_train2 |> pull("cases")
  y <- f(y_obs)
  gp <- mleHomGP(X, y)

  original_data <- data
  filtered_data <- df_train2
  model <- gp
  scaler <- scaler
  return(list(
    original_data = original_data, filtered_data = filtered_data,
    model = model, scaler = scaler, predictors = predictors
  ))
}

predict_model <- function(model_list, pred_data) {
  predictors <- model_list$predictors
  df_test <- pred_data |> dplyr::select("date", "cases", predictors)

  df_test2 <- df_test
  df_test2[predictors] <- predict(model_list$scaler, df_test |> dplyr::select(predictors))

  X <- df_test2 |>
    select(predictors) |>
    as.matrix()
  y_obs <- df_test2 |> pull("cases")
  y <- f(y_obs)

  ppt <- predict(model_list$model, X)

  yyt <- ppt$mean
  q1t <- ppt$mean + qnorm(0.025, 0, sqrt(ppt$sd2 + ppt$nugs)) # lower bound
  q2t <- ppt$mean + qnorm(0.975, 0, sqrt(ppt$sd2 + ppt$nugs)) # upper bound

  # Back transform our data to original
  gp_yy <- fi(yyt)
  gp_q1 <- fi(q1t)
  gp_q2 <- fi(q2t)


  prediction <- tibble(
    x = df_test2 |> pull("date"), y = gp_yy, q1 = gp_q1, q2 = gp_q2,
  )

  rmse <- Metrics::rmse(y, yyt)

  return(list(prediction = prediction, rmse = rmse))
}

get_AIC <- function(model_list) {
  model <- model_list$model
  LL <- model$ll
  n_samples <- nrow(model$X0)
  k <- length(model$theta) + 1 ## only true for HomGP
  AIC <- 2 * k - 2 * LL
  AIC <- AIC + (2 * k**2 + 2 * k) / (n_samples - k - 1)
  return(AIC)
}

forward_selection <- function(data, list_predictors, max_predictors = 5) {
  selected <- c()
  remaining <- list_predictors
  best_aic <- Inf
  results <- list()

  for (i in seq_len(min(max_predictors, length(list_predictors)))) {
    aic_candidates <- sapply(remaining, function(p) {
      predictors <- c(selected, p)
      # print(predictors)
      model_list <- train_model(data, predictors)
      # print(model_list)
      return(get_AIC(model_list))
    }, simplify = TRUE, USE.NAMES = TRUE)
    print(aic_candidates)
    best_candidate <- names(which.min(aic_candidates))
    best_candidate_aic <- min(aic_candidates)
    if (best_candidate_aic - best_aic < -2) {
      selected <- c(selected, best_candidate)
      remaining <- setdiff(remaining, best_candidate)
      best_aic <- best_candidate_aic
      results[[i]] <- list(selected = selected, aic = best_aic)
    } else {
      break
    }
  }
  # results <- NULL
  return(results)
}
