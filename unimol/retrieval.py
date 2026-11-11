#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import torch
from unicore import checkpoint_utils, options
from unicore import tasks

if __package__:
    from .utils.inference import add_device_args, inference_device
else:
    from utils.inference import add_device_args, inference_device

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


@torch.no_grad()
def main(args):
    device = inference_device(args, logger)

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    model.float()
    if args.fp16:
        model.half()
    model.to(device)

    # Print args
    logger.info(args)


    model.eval()
    
    names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)

    # save to ranked_compounds.txt
    with open(os.path.join(args.emb_dir, 'ranked_compounds.txt'), 'w') as f:
        for name, score in zip(names, scores):
            f.write(f"{name}\t{score}\n")


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    add_device_args(parser)
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--emb-dir", type=str, default="", help="path for saved embedding data")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    main(args)


if __name__ == "__main__":
    cli_main()
