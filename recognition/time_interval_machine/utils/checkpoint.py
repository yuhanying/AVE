import torch

from pathlib import Path

import time_interval_machine.utils.logging as logging

logger = logging.get_logger(__name__)

def load_checkpoint(args, model):
    data_parallel = args.num_gpus > 1
    # Load the checkpoint on CPU to avoid GPU mem spike.
    logger.info(F"Loading Model from Path: {args.ckpt_path_best}")
    checkpoint = torch.load(args.ckpt_path_best, map_location="cpu")
    ms = model.module if data_parallel else model

    # Load weights
    pre_train_dict = checkpoint["model"]
    model_dict = ms.state_dict()
    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k
        for k in model_dict.keys()
        if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            print("Network weights {} not loaded.".format(k))
    # Load pre-trained weights.
    ms.load_state_dict(pre_train_dict_match, strict=False)
    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    return epoch, checkpoint

def save_checkpoint(args, state, is_best):
    weights_dir = args.output_dir / Path('models')
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True)

    # filename = f"checkpoint_{state['epoch']}.pth.tar"
    # torch.save(state, weights_dir / filename)
    # logger.info(f"Model Saved to Path: {weights_dir / filename}")
    start_epoch=state['epoch']
    
    if "act_visual" in is_best:
        filename = "model_best_visual.pth"
        torch.save(state, weights_dir / filename)
        logger.info(f"Model Saved to Path: {weights_dir / filename}")
    # else:
    #     logger.info(f"Model No improve after: {state['epoch']}")
        
    if "mt_visual" in is_best:
        filename = "model_best_mt_visual.pth"
        torch.save(state, weights_dir / filename)
        logger.info(f"Model Saved to Path: {weights_dir / filename}")
    # else:
    #     logger.info(f"Model No improve after: {state['epoch']}")

    if "audio" in is_best:
        filename = "model_best_audio.pth"
        torch.save(state, weights_dir / filename)
        logger.info(f"Model Saved to Path: {weights_dir / filename}")
        start_epoch=state['epoch']
    else:
        print(start_epoch)
        print(state["epoch"])
        logger.info(f"Model No improve after: {state['epoch']-start_epoch}")
        
    if "combined" in is_best:
        filename = "model_best_combined.pth"
        torch.save(state, weights_dir / filename)
        logger.info(f"Model Saved to Path: {weights_dir / filename}")
    # else:
    #     logger.info(f"Model No improve after: {state['epoch']}")