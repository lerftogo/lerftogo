import viser
import viser.transforms as tf
import time
import numpy as np
import trimesh as tr
import tyro
import tqdm
import matplotlib

from load_ns_model import NerfstudioWrapper, RealsenseCamera

def main(
    config_path: str  # Nerfstudio model config path, of format outputs/.../config.yml
    ):
    ns_wrapper = NerfstudioWrapper(config_path)
    world_pointcloud, bbox = ns_wrapper.create_pointcloud()

    server = viser.ViserServer()
    server.add_point_cloud(
        name=f"world_pointcloud",
        points=np.asarray(world_pointcloud.vertices),
        colors=np.asarray(world_pointcloud.visual.vertex_colors[:, :3]),
        point_size=0.005,
    )
    
    # Set LERF query configs here
    gen_grasp_text = server.add_gui_text(
        name=f"LERF query",
        initial_value="",
    )
    gen_lerf_threshold_slider = server.add_gui_slider(
        name=f"LERF threshold",
        min=0.0,
        max=1.0,
        initial_value=0.5,
        step=0.01
    )
    query_pointcloud_button = server.add_gui_button(
        name=f"Query LERF",
    )
    create_grasp_button = server.add_gui_button(
        name=f"Create Grasps",
    )
    ns_wrapper.pipeline.model.clip_scales.value = True

    relevancy_info = {
        "relevancy": None,
        "xyz": None,
    }
    @query_pointcloud_button.on_click
    def _(_):
        gen_lerf_threshold_slider.disabled = True
        query_pointcloud_button.disabled = True
        gen_grasp_text.disabled = True

        lerf_word = gen_grasp_text.value.split(";")
        ns_wrapper.pipeline.image_encoder.set_positives(lerf_word)

        relevancy_dict = {}
        for ind in tqdm.trange(ns_wrapper.num_cameras//2, ns_wrapper.num_cameras, 4):
            c2w = ns_wrapper.get_train_camera_c2w(ind)
            rscam = RealsenseCamera.get_camera(ns_wrapper.visercam_to_ns(c2w), downscale=1/4)

            outputs = ns_wrapper(camera=rscam, render_lerf=True)
            relevancy = outputs[f'relevancy_{len(lerf_word)-1}']
            relevancy_dict[ind] = (relevancy, outputs['xyz'])

        relevancy_info['relevancy'] = np.stack([relevancy_dict[ind][0] for ind in range(ns_wrapper.num_cameras//2, ns_wrapper.num_cameras, 4)]) # relevancy
        relevancy_info['xyz'] = np.stack([relevancy_dict[ind][1] for ind in range(ns_wrapper.num_cameras//2, ns_wrapper.num_cameras, 4)]) # xyz

        gen_lerf_threshold_slider.disabled = False
        query_pointcloud_button.disabled = False
        gen_grasp_text.disabled = False

    @gen_lerf_threshold_slider.on_update
    def _(_):
        if relevancy_info['relevancy'] is None:
            return
        if 'viser_pc' in relevancy_info:
            relevancy_info['viser_pc'].remove()
        mask = relevancy_info['relevancy'] > gen_lerf_threshold_slider.value
        
        if mask.sum() == 0:
            return

        colors = relevancy_info['relevancy'][mask]
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        colors = matplotlib.colormaps['viridis'](colors)[:, :3]
        relevancy_info['viser_pc'] = server.add_point_cloud(
            name=f"lerf_pointcloud",
            points=relevancy_info['xyz'][mask],
            colors=colors,
            point_size=0.005,
        )


    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    tyro.cli(main)