from pathlib import Path
import os.path as osp
import time

import nerfstudio.configs.method_configs as method_configs

config = method_configs.all_methods['lerf-lite']
data = Path('output/scene_1_utensils/')
config.pipeline.datamanager.data = data
config.max_num_iterations = 10 # 2000
config.steps_per_save = config.max_num_iterations
config.timestamp = "one_model"
config.viewer.quit_on_train_completion = True

if (
    osp.exists(config.get_base_dir()) and
    osp.exists(config.get_base_dir() / "nerfstudio_models")
):
    config.load_dir = config.get_base_dir() / "nerfstudio_models"
    config.load_step = 10
    print("we are going to load a model")
else:
    print("we are going to train a model")

output_config = config.get_base_dir() / 'config.yaml'
trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()

start = time.time()
trainer.train()
print(f"Training took {time.time() - start} seconds")

import pdb; pdb.set_trace()
print()