#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test an audio classification model."""
import os,sys

# 取得目前的工作目錄 (預期是 tim-main)
current_dir = os.getcwd()
print("Current Working Directory:", current_dir)


feature_extractors_dir = os.path.join(current_dir, "feature_extractors/auditory_slowfast")
os.chdir(feature_extractors_dir)

print("New Working Directory:", os.getcwd())

sys.path.append(os.getcwd())
print("sys.path:", sys.path)

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform multi-clip testing.
    launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
