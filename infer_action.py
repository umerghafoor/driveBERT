import torch
import argparse
import numpy as np
from lib.model.model_action import ActionNet
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone, load_pretrained_weights

from collections import OrderedDict

def load_sample(npy_path):
    data = np.load(npy_path)  # (T, 17, 3)
    data = data[:243]  # trim to 243 frames
    data = data[np.newaxis, np.newaxis, :, :, :]  # (1, 1, 243, 17, 3)
    data = np.repeat(data, 2, axis=1)  # (1, 2, 243, 17, 3)
    return torch.tensor(data).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input .npy file')
    args = parser.parse_args()

    cfg = get_config(args.config)

    # Load backbone and model
    backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=backbone,
        dim_rep=cfg.dim_rep,
        num_classes=cfg.action_classes,
        dropout_ratio=cfg.dropout_ratio,
        version=cfg.model_version,
        hidden_dim=cfg.hidden_dim,
        num_joints=cfg.num_joints
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint['model'] = OrderedDict(
        (k.replace("module.", ""), v) for k, v in checkpoint['model'].items()
    )
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()


    if torch.cuda.is_available():
        model = model.cuda()

    # Load input tensor
    input_tensor = load_sample(args.input)

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        print(f"Predicted action class: {pred.item()}")
        print(f"Predicted action class (one-hot): {pred}")
        for idx, prob in enumerate(probabilities[0]):
            print(f"Class {idx}: {prob.item():.4f}")

if __name__ == "__main__":
    main()
