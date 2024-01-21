from configs.run_config import config

hyperparameters = {
  "learning_rate": 0.0003,
  "buffer_size": 1000000,
  "learning_starts": 100,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "train_freq": 1,
  "gradient_steps": 1,
  "action_noise": None,
  "replay_buffer_class": None,
  "replay_buffer_kwargs": None,
  "optimize_memory_usage": False,
  "ent_coef": 'auto',
  "target_update_interval": 1,
  "target_entropy": 'auto',
  "use_sde": False,
  "sde_sample_freq": -1,
  "use_sde_at_warmup": False,
  "stats_window_size": 100,
  "tensorboard_log": "logs_sb3/SAC",
  "policy_kwargs": None,
  "verbose": 1,
  "seed": config["seed"],
  "device": 'auto',
  # " _init_setup_model": True,

  "num_episodes": 200
}
