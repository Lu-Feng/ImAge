
import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))

def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, weights_only=False)
    start_epoch_num = checkpoint["epoch_num"]+1
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r1_r5 = checkpoint["best_r1_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current best (R@1 + R@5) = {best_r1_r5:.1f}")
    if args.resume.endswith("best_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("best_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r1_r5, start_epoch_num, not_improved_num
