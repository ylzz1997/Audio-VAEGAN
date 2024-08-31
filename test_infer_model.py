import torch
import torchaudio
import os
import argparse
import json
from modules.models import Generator, Encoder


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="the directory of the inference models",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output audio file",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # load model
    args = parse_args()
    config_path = os.path.join(args.model, "config.json")
    enc_path = os.path.join(args.model, "encoder.pth")
    dec_path = os.path.join(args.model, "decoder.pth")
    # load config
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    # load model
    enc = Encoder(h=config["hps"]).cuda()
    dec = Generator(h=config["hps"]).cuda()
    enc.load_state_dict(torch.load(enc_path, weights_only=False))
    dec.load_state_dict(torch.load(dec_path, weights_only=False))
    enc.eval()
    dec.eval()
    enc.remove_weight_norm()
    dec.remove_weight_norm()
    # load audio
    waveform, sr = torchaudio.load(args.input)
    # inference
    with torch.no_grad():
        z, m, logs = enc(waveform.cuda())
        output = dec(z).squeeze(0).cpu()
    # save audio
    print(output.shape)
    torchaudio.save(args.output, output, sr)
