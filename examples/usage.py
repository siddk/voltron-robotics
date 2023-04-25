"""
usage.py

Example script demonstrating how to load a Voltron model (`V-Cond`) and instantiate a Multiheaded Attention Pooling
extractor head for downstream tasks.

This is the basic formula/protocol for using Voltron for arbitrary downstream applications.

Run with (from root of repository): `python examples/usage.py`
"""
import torch
from torchvision.io import read_image

from voltron import instantiate_extractor, load


def usage() -> None:
    print("[*] Demonstrating Voltron Usage for Various Adaptation Applications")

    # Get `torch.device` for loading model (note -- we'll load weights directly onto device!)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Voltron model --> specify `freeze`, `device` and get model (nn.Module) and preprocessor
    vcond, preprocess = load("v-cond", device=device, freeze=True)

    # Obtain and preprocess an image =>> can be from a dataset, from a camera on a robot, etc.
    img = preprocess(read_image("examples/img/peel-carrot-initial.png"))[None, ...].to(device)
    lang = ["peeling a carrot"]

    # Get various representations...
    with torch.no_grad():
        multimodal_features = vcond(img, lang, mode="multimodal")  # Fused vision & language features
        visual_features = vcond(img, mode="visual")  # Vision-only features (no language)

    # Can instantiate various extractors for downstream applications
    vector_extractor = instantiate_extractor(vcond, n_latents=1, device=device)()
    seq_extractor = instantiate_extractor(vcond, n_latents=64, device=device)()

    # Assertions...
    assert list(vector_extractor(multimodal_features).shape) == [1, vcond.embed_dim], "Should return a dense vector!"
    assert list(seq_extractor(visual_features).shape) == [1, 64, vcond.embed_dim], "Should return a sequence!"


if __name__ == "__main__":
    usage()
