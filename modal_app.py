import os
import sys
import pickle
import numpy as np

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

        print(f"PATH_REPO: {PATH_REPO}")

        ckpt_filename = "ng.pt"
        ckpt_path = f"/data/{ckpt_filename}"
        
        if not os.path.exists(ckpt_path):
            print(f"Downloading {ckpt_filename} to {ckpt_path}...")
            # We download to a temp location and move to volume to be safe/atomic-ish? 
            # Or just download directly.
            try:
                hf_hub_download(repo_id="nvidia/NitroGen", filename=ckpt_filename, local_dir="/data")
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
                raise e
        
        print("Loading model...")
        # Load model logic adapted from InferenceSession.from_ckpt to avoid input()
        model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(ckpt_path)
        
        # Default game selection (Unconditional)
        # You could also use an environment variable or file in the volume to persist selection
        selected_game = None 
        
        # If you want to enable a specific game by default, you could map it here:
        # if game_mapping and "SomeGame" in game_mapping:
        #     selected_game = "SomeGame"

        print(f"Model loaded. Selected game: {selected_game}")

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
        return self.session.predict(image)

    @modal.method()
    def reset(self):
        self.session.reset()
        return "ok"

    @modal.method()
    def info(self):
        return self.session.info()
