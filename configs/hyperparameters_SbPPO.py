from configs.run_config import config

hyperparameters = {
  "learning_rate": 0.0003,
  "horizon": 2048,
  "batch_size": 64,
  "num_epochs": 10,
  "discount_factor": 0.99,
  "gae_lambda": 0.95,
  "epsilon": 0.2,
  "clip_range_vf": None,
  "normalize_advantage": True,
  "ent_coef": 0.0,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "use_sde": False,
  "sde_sample_freq": -1,
  "target_kl": None,
  "stats_window_size": 100,
  "tensorboard_log": "logs_sb3/PPO",
  "policy_kwargs": None,
  "verbose": 1,
  "seed": config["seed"],
  "device": 'auto',
  # " _init_setup_model": True,

  "num_episodes": 200
}
