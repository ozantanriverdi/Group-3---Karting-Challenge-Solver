hyperparameters = {
  "learning_rate": 0.0003,
  "discount_factor": 0.99,
  "epsilon": 0.1,
  "gae_lambda": 0.95,
  "value_function_coef": 0.5, # not used yet
  "num_episodes": 300,
  "num_epochs": 10,
  "batch_size": 64,
  "horizon": 512,
  "entropy_coefficient_init": 0.1,
  "entropy_coefficient_target": 0.01,
  "entropy_coefficient_const": 0.01,
  "entropy_decay_steps": 100000
}
