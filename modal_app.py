import os
import sys
import pickle
import numpy as np
import time
import traceback

import modal

# Define the Modal image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "numpy",
        "pyzmq",
        "torch",
        "pyyaml",
        "einops",
        "transformers",
        "pydantic>=2.0",
        "diffusers",
        "polars",
        "huggingface_hub",
        "opencv-python-headless",
        "pillow",
        "accelerate" # Often needed for transformers/diffusers
    )
    .add_local_dir(".", remote_path="/root/NitroGen")
)

app = modal.App("nitrogen-inference", image=image)
volume = modal.Volume.from_name("nitrogen-vol", create_if_missing=True)

@app.cls(
    gpu="A100",  # Or A100/A10G depending on availability/needs. H100 is fastest.
    timeout=600, 
    volumes={"/data": volume}, 
)
class Model:
    @modal.enter()
    def load(self):
        # Add the root directory to sys.path so we can import nitrogen
        sys.path.append("/root/NitroGen")
        
        # Import here to avoid issues if local env doesn't have deps
        from nitrogen.inference_session import load_model, InferenceSession
        from nitrogen.shared import PATH_REPO
        from huggingface_hub import hf_hub_download
        import torch

        print("[modal] booting NitroGen model")
        print(f"[modal] python={sys.version}")
        print(f"[modal] PATH_REPO={PATH_REPO}")
        print(f"[modal] torch={getattr(torch, '__version__', 'unknown')} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"[modal] cuda_device={torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"[modal] cuda device query failed: {e}")

        ckpt_filename = "ng.pt"
        ckpt_path = f"/data/{ckpt_filename}"
        print(f"[modal] checkpoint path={ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            print(f"[modal] downloading {ckpt_filename} to {ckpt_path}...")
            # We download to a temp location and move to volume to be safe/atomic-ish? 
            # Or just download directly.
            try:
                t0 = time.time()
                hf_hub_download(repo_id="nvidia/NitroGen", filename=ckpt_filename, local_dir="/data")
                print(f"[modal] download complete. latency={time.time() - t0:.3f}s")
            except Exception as e:
                print(f"[modal] error downloading checkpoint: {e}")
                print(traceback.format_exc())
                raise
        
        print("[modal] loading model...")
        # Load model logic adapted from InferenceSession.from_ckpt to avoid input()
        t0 = time.time()
        model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(ckpt_path)
        print(f"[modal] model load complete. latency={time.time() - t0:.3f}s")
        
        # Default game selection (Unconditional)
        # You could also use an environment variable or file in the volume to persist selection
        selected_game = None 
        
        # If you want to enable a specific game by default, you could map it here:
        # if game_mapping and "SomeGame" in game_mapping:
        #     selected_game = "SomeGame"

        print(f"[modal] model loaded. selected_game={selected_game}")

        self.session = InferenceSession(
            model=model,
            ckpt_path=ckpt_path,
            tokenizer=tokenizer,
            img_proc=img_proc,
            ckpt_config=ckpt_config,
            game_mapping=game_mapping,
            selected_game=selected_game,
            old_layout=False,
            cfg_scale=1.0,
            action_downsample_ratio=action_downsample_ratio,
            context_length=1
        )

    @modal.method()
    def predict(self, image):
        # Image is expected to be a numpy array or compatible
        t0 = time.time()
        try:
            if isinstance(image, np.ndarray):
                print(f"[modal] predict image shape={image.shape} dtype={image.dtype} nbytes={image.nbytes}")
            else:
                print(f"[modal] predict image type={type(image)}")
            out = self.session.predict(image)
            print(f"[modal] predict complete. latency={time.time() - t0:.3f}s")
            return out
        except Exception as e:
            print(f"[modal] predict error: {e}")
            print(traceback.format_exc())
            raise

    @modal.method()
    def reset(self):
        t0 = time.time()
        try:
            self.session.reset()
            print(f"[modal] reset complete. latency={time.time() - t0:.3f}s")
            return "ok"
        except Exception as e:
            print(f"[modal] reset error: {e}")
            print(traceback.format_exc())
            raise

    @modal.method()
    def info(self):
        t0 = time.time()
        try:
            info = self.session.info()
            print(f"[modal] info latency={time.time() - t0:.3f}s")
            return info
        except Exception as e:
            print(f"[modal] info error: {e}")
            print(traceback.format_exc())
            raise
