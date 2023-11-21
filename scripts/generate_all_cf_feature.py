import subprocess, os
from subprocess import Popen

my_env = os.environ.copy()

scene_names = ["scene_8_take_1",
               "martini_test",
               "flower_figure"]

# scene_names = ["scene_2_take_1", 
#                "scene_2_take_2", 
#                "scene_3_take_1"]

# scene_names = [ "scene_4_take_1", 
#                "scene_5_take_1", 
#                "scene_6_take_1" ]

# scene_names = ["scene_7_take_1", 
#                "scene_8_take_1",
#                "martini_test",
#                "flower_figure"]

for scene_name in scene_names:
    os.system(f"python ../conceptfusion_baseline/load_depths.py --config-path outputs/{scene_name}/lerf-lite/one_model/config.yml --transform output/{scene_name}/transforms.json --savedir cf/{scene_name}")
    os.system(f"python ../conceptfusion_baseline/concept-fusion/examples/extract_conceptfusion_features.py --data-dir cf --sequence {scene_name} --savedir cf/{scene_name}")
    os.system(f"python ../conceptfusion_baseline/concept-fusion/examples/run_feature_fusion_and_save_map.py --data-dir cf --sequence {scene_name} --savedir cf/{scene_name}")
