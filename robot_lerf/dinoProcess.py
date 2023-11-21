import multiprocessing as mp
import torch
import torchvision.transforms as transforms
import queue
import time
import numpy as np
from lerf.data.utils.dino_extractor import ViTExtractor

class DinoProcess(mp.Process):
    def __init__(self, out_queue, pca_dim=64, dino_model_type="dino_vits8", dino_load_size = 224, dino_layer = 11, dino_facet = "key", dino_bin = False):
        super().__init__()
        self.out_queue = out_queue
        self.in_queue = mp.Queue(maxsize=0)
        self.device = "cuda:0"
        self.pca_dim = pca_dim
        self.dino_model_type = dino_model_type
        self.dino_load_size = dino_load_size
        self.dino_layer = dino_layer
        self.dino_facet = dino_facet
        self.dino_bin = dino_bin
        self.im_count = 0
        self.daemon=True


    def run(self):
        if self.dino_model_type == "dino_vits8":
            self.dino_stride = 4
        elif self.dino_model_type == "dino_vitb8":
            self.dino_stride = 4
        elif self.dino_model_type == "dinov2_vitb14":
            self.dino_stride = 14
        else:
            raise NotImplementedError()
        extractor = ViTExtractor(self.dino_model_type, self.dino_stride,device=self.device)

        running=True
        while running:
            img_batch = []
            while True:
                try:
                    img = self.in_queue.get(timeout=.01)
                except queue.Empty:
                    break
                if img is None:
                    running = False
                    print("DINO DONE")
                    break
                img = torch.from_numpy(img).to(self.device).float() / 255.0
                img = torch.permute(img, (2, 0, 1))
                img_batch.append(img)
            if len(img_batch) == 0:
                continue

            start=time.time()
            img_batch = torch.stack(img_batch)
            img_batch = extractor.preprocess(img_batch, self.dino_load_size)[0].to(self.device)
            if len(img_batch.shape) == 3:
                img_batch = img_batch.unsqueeze(0)
            for image in img_batch:
                self.im_count += 1
                # image nees to be resized s.t. H, W are divisible by dino_stride
                if "dinov2" in self.dino_model_type:
                    image = transforms.Resize(
                        (
                            (image.shape[1] // self.dino_stride) * self.dino_stride,
                            (image.shape[2] // self.dino_stride) * self.dino_stride,
                        )
                    )(image)
                with torch.no_grad():
                    descriptors = extractor.extract_descriptors(
                        image.unsqueeze(0),
                        [self.dino_layer],
                        self.dino_facet,
                        self.dino_bin,
                    )
                descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
                self.out_queue.put(descriptors.cpu().numpy())
            print(f"Dino took {time.time()-start} seconds")

    def kill(self):
        self.in_queue.put(None)


# if __name__=="__main__":
#     mp.set_start_method('spawn')
#     dino_q = mp.Queue()
#     dino_p = DinoProcess(dino_q)
#     dino_p.start()
#     input("Enter to continue")
#     i = 0
#     while i < 50:
#         dummy_image = np.random.randint(10, size=(100, 200, 3))
#         print(f"Image_{i}")
#         dino_p.in_queue.put(dummy_image)
#         time.sleep(.1)
#         i += 1

#     dino_p.kill()
#     dino_p.in_queue.close()
#     dino_p.in_queue.join_thread()

#     time.sleep(2)
#     for j in range(i):
#         print(j)
#         dino_batch = []
#         dino_batch.append(dino_q.get())
#     print("Done")