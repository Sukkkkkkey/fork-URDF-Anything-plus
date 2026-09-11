#!/usr/bin/env python3
"""Apply the TripoSG encoder import required by URDF-Anything+ inference."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triposg-root", type=Path, default=Path("TripoSG"))
    args = parser.parse_args()
    target = (
        args.triposg_root
        / "triposg/models/autoencoders/autoencoder_kl_triposg.py"
    )
    text = target.read_text(encoding="utf-8")
    enabled = "from torch_cluster import fps"
    disabled = "# from torch_cluster import fps"
    if disabled in text:
        target.write_text(text.replace(disabled, enabled, 1), encoding="utf-8")
        print(f"patched {target}")
    elif enabled in text:
        print(f"already patched {target}")
    else:
        raise RuntimeError(f"expected torch_cluster import not found in {target}")


if __name__ == "__main__":
    main()
