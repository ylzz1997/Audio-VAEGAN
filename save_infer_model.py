import torch
import os
import json
import argparse


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config.json of the model",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="directory to save the inference model",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == "__main__":
    args = parse_args()
    # read config file
    with open(args.config, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    # transfer config
    hop_size = config["data"]["hop_length"]
    windows_size = config["data"]["win_length"]
    inter_channels = config["model"]["inter_channels"]
    resblock = config["model"]["resblock"]
    resblock_kernel_sizes = config["model"]["resblock_kernel_sizes"]
    resblock_dilation_sizes = config["model"]["resblock_dilation_sizes"]
    upsample_rates = config["model"]["upsample_rates"]
    upsample_initial_channel = config["model"]["upsample_initial_channel"]
    upsample_kernel_sizes = config["model"]["upsample_kernel_sizes"]
    try:
        sampling_rate = config["model"]["sampling_rate"]
    except KeyError:
        sampling_rate = 44100
    try:
        use_vq = config["model"]["use_vq"]
    except KeyError:
        use_vq = False
    try:
        codebook_size = config["model"]["codebook_size"]
    except KeyError:
        codebook_size = 4096
    hps = {
        "sampling_rate": sampling_rate,
        "inter_channels": inter_channels,
        "resblock": resblock,
        "resblock_kernel_sizes": resblock_kernel_sizes,
        "resblock_dilation_sizes": resblock_dilation_sizes,
        "upsample_rates": upsample_rates,
        "upsample_initial_channel": upsample_initial_channel,
        "upsample_kernel_sizes": upsample_kernel_sizes
    }
    save_config = {
        "hop_size": hop_size,
        "windows_size": windows_size,
        "hps": hps,
        "use_vq": use_vq,
        "codebook_size": codebook_size
    }
    # load checkpoint
    model = torch.load(args.model, map_location="cpu", weights_only=True)
    model = model["model"]
    save_model_enc = {}
    save_model_dec = {}
    for k, v in model.items():
        if k[:4] == 'dec.':
            k = k[len(k.split('.')[0]) + 1:]
            save_model_dec[k] = v
        else:
            k = k[len(k.split('.')[0]) + 1:]
            save_model_enc[k] = v
    # save inference model
    os.makedirs(args.output, exist_ok=True)
    torch.save(save_model_enc, os.path.join(args.output, "encoder.pth"))
    torch.save(save_model_dec, os.path.join(args.output, "decoder.pth"))
    # save config
    with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
        json.dump(save_config, f, indent=4)
