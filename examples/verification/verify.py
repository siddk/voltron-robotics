"""
verify.py

Example script demonstrating how to load all Voltron models (and reproduced models), take input image(s), and get the
various (e.g., multimodal, image-only) representations.

Also serves to verify that representation loading is working as advertised.

Run with (from root of repository): `python examples/verification/verify.py`
"""
import torch
from torchvision.io import read_image

from voltron import load

# Available Models
MODELS = ["v-cond", "v-dual", "v-gen", "r-mvp", "r-r3m-vit", "r-r3m-rn50"]

# Sample Inputs
IMG_A, IMG_B = "examples/verification/img/peel-carrot-initial.png", "examples/verification/img/peel-carrot-final.png"
LANGUAGE = "peeling a carrot"


def verify() -> None:
    print("[*] Running `verify` =>> Verifying Model Representations!")

    # Read both images (we'll use the second image for the dual-frame models)
    image_a, image_b = read_image(IMG_A), read_image(IMG_B)

    # Get `torch.device` for loading model (note -- we'll load weights directly onto device!)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_id in MODELS:
        print(f"\t=> Loading Model ID `{model_id}` and Verifying Representation Shapes!")
        model, preprocess = load(model_id, device=device, freeze=True)

        # Preprocess image, run feature extraction --> assert on shapes!
        if model_id in {"v-cond", "v-cond-base"}:
            for modality, expected in [("multimodal", 196 + 20), ("visual", 196)]:
                representation = model(preprocess(image_a)[None, ...].to(device), [LANGUAGE], mode=modality)
                assert representation.squeeze(dim=0).shape[0] == expected, "Shape not expected!"

        elif model_id in {"v-dual", "v-gen"}:
            for modality, expected in [("multimodal", 196 + 20), ("visual", 196)]:
                dual_img = torch.stack([preprocess(image_a), preprocess(image_b)])[None, ...].to(device)
                representation = model(dual_img, [LANGUAGE], mode=modality)
                assert representation.squeeze(dim=0).shape[0] == expected, "Shape not expected!"

        elif model_id == "r-mvp":
            for mode, expected in [("patch", 196), ("cls", 1)]:
                representation = model(preprocess(image_a)[None, ...].to(device), mode=mode)
                assert representation.squeeze(dim=0).shape[0] == expected, "Shape not expected!"

        elif model_id in {"r-r3m-vit", "r-r3m-rn50"}:
            representation = model(preprocess(image_a)[None, ...].to(device))
            assert representation.squeeze(dim=0).shape[0] == 1, "Shape not expected!"

        else:
            raise ValueError(f"Model {model_id} not supported!")

    # We're good!
    print("[*] All representations & shapes verified! Yay!")


if __name__ == "__main__":
    verify()
