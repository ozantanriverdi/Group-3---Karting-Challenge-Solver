from configs.run_config import config

hyperparameters = {
  "learning_rate": 0.0007,
  "n_steps": 5,
  "discount_factor": 0.99,
  "gae_lambda": 1.0,
  "ent_coef": 0.0,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "rms_prop_eps": 1e-05,
  "use_rms_prop": True,
  "use_sde": False,
  "sde_sample_freq": -1,
  "normalize_advantage": False,
  "stats_window_size": 100,
  "tensorboard_log": "logs_sb3/A2C",
  "policy_kwargs": None,
  "verbose": 1,
  "seed": config["seed"],
  "device": 'auto',
  # " _init_setup_model": True,

  "num_episodes": 200
}
