library(tidymodels)
library(dplyr)
library(readr)

train <- read_csv("train.csv")
test <- read_csv("test.csv")

# feature engineering 
train <- train |> 
  mutate(
    # Distance differences
    Hydro_Road_diff = Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways,
    Fire_Hydro_diff = Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Hydrology,
    Fire_Road_diff  = Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways,
    
    # Distance sums
    Hydro_Road_sum = Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Roadways,
    Fire_Hydro_sum = Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Hydrology,
    Fire_Road_sum  = Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Roadways
  )

test <- test |> 
  mutate(
    Hydro_Road_diff = Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways,
    Fire_Hydro_diff = Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Hydrology,
    Fire_Road_diff  = Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways,
    
    Hydro_Road_sum = Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Roadways,
    Fire_Hydro_sum = Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Hydrology,
    Fire_Road_sum  = Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Roadways
  )

train <- train %>%
  mutate(
    Cover_Type = as.factor(Cover_Type)
  )


forest_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  update_role(Id, new_role = "ID") %>%
  step_zv(all_predictors())   # remove zero-variance predictors

rf_model <- rand_forest(
  mode = "classification",
  trees = 100,     # <-- much faster during tuning
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", splitrule = "extratrees")


set.seed(4)
cv_splits <- vfold_cv(train, v = 5)

rf_grid <- grid_random(
  parameters(list(
    mtry(range = c(5, 50)),
    min_n(range = c(1, 20))
  )),
  size = 30
)


rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(forest_recipe)

rf_tune <- tune_grid(
  rf_workflow,
  resamples = cv_splits,
  grid = rf_grid,
  metrics = metric_set(accuracy),
  control = control_grid(verbose = TRUE)
)


best_mtry <- select_best(rf_tune, metric = "accuracy")

final_rf <- finalize_workflow(rf_workflow, best_mtry) %>%
  update_model(
    rand_forest(
      mode = "classification",
      trees = 500,
      mtry = best_mtry$mtry,
      min_n = best_mtry$min_n
    ) %>% 
      set_engine("ranger", splitrule = "extratrees")
  ) %>%
  fit(data = train)




predictions <- predict(final_rf, test, type = "class") %>%
  bind_cols(test %>% select(Id)) %>%
  rename(Cover_Type = .pred_class) %>%
  mutate(Id = as.integer(Id))  # <-- force integer

# Write CSV without scientific notation
write.csv(predictions, "submission.csv", row.names = FALSE, quote = FALSE)
