import torch
from lerf.encoders.openclip_encoder import OpenCLIPNetwork,OpenCLIPNetworkConfig
from lerf.data.lerf_datamanager import LERFDataManagerConfig
import multiprocessing as mp
import queue
from typing import List,Tuple,Dict
import json
import numpy as np

class ClipProcess(mp.Process):
    def __init__(self, out_queue, device):
        super().__init__()
        self.out_queue = out_queue
        self.in_queue = mp.Queue(maxsize=0)
        self.device = device
        self.daemon=True
        self.pyramid_config:LERFDataManagerConfig = LERFDataManagerConfig()
        self.img_shape = [896,1600]
        self.tile_sizes = torch.linspace(*self.pyramid_config.patch_tile_size_range, self.pyramid_config.patch_tile_size_res).to(self.device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), self.pyramid_config.patch_stride_scaler) for tr in self.tile_sizes]

    def run(self):
        self.model = OpenCLIPNetworkConfig(device=self.device).setup()
        while True:
            try:
                img = self.in_queue.get(timeout=.01)
            except queue.Empty:
                continue
            if img is None:
                print("CLIP DONE")
                break
            print("Processing clip")
            #now process the image at all scales
            #reformat to tensor
            img = torch.from_numpy(img).to(self.device).permute(2,0,1).float() / 255.0
            scale_embeddings = self._get_all_scales(img)
            #convert them to numpy 
            scale_embeddings = [s.numpy() for s in scale_embeddings]
            self.out_queue.put(scale_embeddings)

    def kill(self):
        self.in_queue.put(None)

    def get_level_json(self,i) -> str:
        """
        need to return tile_ratio, stride_ratio, image_shape, model_name
        """
        return json.dumps({"tile_ratio":self.tile_sizes[i].item(),"stride_ratio": self.strider_scaler_list[i],
                     "image_shape":self.img_shape,"model_name":self.clip_model_name()})

    def get_pyramid_json(self):
        """
        tile_size_range, tile_size_res, stride_scaler, image_shape, model_name
        """
        return json.dumps({"tile_size_range":self.pyramid_config.patch_tile_size_range,"tile_size_res":self.pyramid_config.patch_tile_size_res,
                    "stride_scaler":self.pyramid_config.patch_stride_scaler,"image_shape":self.img_shape,"model_name":self.clip_model_name()})

    def clip_model_name(self):
        return "openclip_{}_{}".format(OpenCLIPNetworkConfig().clip_model_type, OpenCLIPNetworkConfig().clip_model_pretrained)
    
    def _get_level_params(self,i):
        stride_scaler = self.strider_scaler_list[i]
        kernel_size = int(self.img_shape[0] * self.tile_sizes[i])
        stride = int(kernel_size * stride_scaler)
        padding = kernel_size // 2
        return kernel_size,stride,padding

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def _get_all_scales(self,img:torch.tensor) -> Tuple[List[torch.tensor],Dict]:
        """
        given an image, return all the scale embeddings
        """
        all_embeds = []
        for i in range(self.pyramid_config.patch_tile_size_res):
            embeds = self._get_one_scale(img,i)
            all_embeds.append(embeds.cpu())
        return all_embeds

    def _get_one_scale(self,img,i:int):
        """
        given an image and scale, return the clip embeddings for the image
        returns 1, nx, ny, 512 embeddings for the image at the given scale
        """
        kernel_size,stride,padding = self._get_level_params(i)
        unfold_func = torch.nn.Unfold(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        center_x = (
            (kernel_size - 1) / 2
            - padding
            + stride
            * np.arange(
                np.floor((self.img_shape[0] + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            )
        )
        center_y = (
            (kernel_size - 1) / 2
            - padding
            + stride
            * np.arange(
                np.floor((self.img_shape[1] + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            )
        )
        aug_imgs = img.to(self.device).unsqueeze(0)
        tiles = unfold_func(aug_imgs).permute(2, 0, 1).view(-1, 3, kernel_size, kernel_size)
        with torch.no_grad():
            clip_embeds = self.model.encode_image(tiles)
            clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        clip_embeds = clip_embeds.reshape((center_x.shape[0], center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds


if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue()
    clipproc = ClipProcess(q,'cuda:0')
    clipproc.start()
    print(clipproc.get_pyramid_json())
    clipproc.join()