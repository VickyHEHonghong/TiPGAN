#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import datetime
import logging
import math
import os

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (DistributedDataParallelKwargs,
                              ProjectConfiguration, set_seed)
from omegaconf import OmegaConf
from packaging import version
from torchvision.transforms import CenterCrop, RandomResizedCrop
from tqdm.auto import tqdm

from src.dataset import HalfDataset
from src.loss import GANLoss, HistogramLoss, StyleLoss
from src.network import Discriminator, ResnetNetGenerator

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description='Simple example of a MVTON training script.')
    parser.add_argument(
        '--config',
        type=str,
        default='baseline.yaml',
        help='path of the yaml config file.',
    )
    parser.add_argument(
        '--scale_lr',
        action='store_true',
        default=False,
        help='Scale the learning rate by the number of GPUs, gradient'
        ' accumulation steps, and batch size.',
    )
    parser.add_argument(
        '--allow_tf32',
        action='store_true',
        help=('Whether or not to allow TF32 on Ampere GPUs. Can be used to '
              'speed up training.'),
    )
    parser.add_argument(
        '--report_to',
        type=str,
        default='tensorboard',
        help=('The integration to report the results and logs to. Supported'
              ' platforms are `"tensorboard"` (default), `"wandb"` and'
              ' `"comet_ml"`. Use `"all"` to report to all integrations.'),
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        help=(
            'The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        '--set_grads_to_none',
        action='store_true',
        help=(
            'Save more memory by using setting grads to None instead of zero.'
            ' Be aware, that this changes certain behaviors, so disable this'
            ' argument if itcauses any problems.'),
    )
    parser.add_argument(
        '--tracker_project_name',
        type=str,
        default='train_tipgan',
        help=(
            'The `project_name` argument passed to Accelerator.init_trackers'),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    cfg = OmegaConf.load(args.config)
    train_cfg = cfg.train
    optim_cfg = cfg.optimizer

    if torch.backends.mps.is_available(
    ) and train_cfg.mixed_precision == 'bf16':
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            'Mixed precision training with bfloat16 is not supported on MPS. '
            'Please use fp16 (recommended) or fp32 instead.')

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if train_cfg.resume_from_checkpoint is None:
        train_cfg.output_dir = os.path.join(train_cfg.output_dir, timestamp)
    else:
        resume_timestamp = train_cfg.resume_from_checkpoint.split('/')[-2]
        train_cfg.output_dir = os.path.join(
            train_cfg.output_dir, f'{timestamp}-resumed_'
            f'from-{resume_timestamp}')
    logging_dir = os.path.join(train_cfg.output_dir, 'logs')

    accelerator_project_config = ProjectConfiguration(
        project_dir=train_cfg.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        mixed_precision=train_cfg.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if train_cfg.seed is not None:
        set_seed(train_cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if train_cfg.output_dir is not None:
            os.makedirs(train_cfg.output_dir, exist_ok=True)

    # create models
    generator = ResnetNetGenerator(input_nc=3,
                                   output_nc=3,
                                   ngf=64,
                                   norm_layer=nn.InstanceNorm2d,
                                   use_dropout=False,
                                   n_blocks=6,
                                   padding_type='reflect')
    discriminator = Discriminator(input_nc=6,
                                  ndf=64,
                                  norm_layer=nn.InstanceNorm2d,
                                  use_sigmoid=True)

    loss_gan = GANLoss(use_lsgan=False,
                       tensor=torch.FloatTensor,
                       target_real_label=1.0,
                       target_fake_label=0.0)
    loss_l1 = torch.nn.L1Loss()
    loss_hist = HistogramLoss()
    loss_style = StyleLoss()

    # For mixed precision training we cast all non-trainable weights (vae,
    # non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full
    # precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            'Mixed precision training with bfloat16 is not supported on MPS.'
            ' Please use fp16 (recommended) or fp32 instead.')

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):
        # create custom saving & loading hooks so that
        # `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                for i in reversed(range(len(models))):
                    model = models[i]
                    weight = weights.pop()

                    # save the model
                    sub_dir = model.__class__.__name__.lower()
                    if 'loss' in sub_dir:
                        continue
                    if hasattr(model, 'save_pretrained'):
                        model.save_pretrained(os.path.join(
                            output_dir, sub_dir))
                    else:
                        os.makedirs(os.path.join(output_dir, sub_dir),
                                    exist_ok=True)
                        torch.save(
                            weight,
                            os.path.join(output_dir, sub_dir,
                                         'state_dict.pth'))

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                sub_dir = model.__class__.__name__.lower()
                if 'loss' in sub_dir:
                    continue

                if hasattr(model, 'load_pretrained'):
                    model.load_pretrained(os.path.join(input_dir, sub_dir))
                elif hasattr(model, 'from_pretrained'):
                    model.from_pretrained(os.path.join(input_dir, sub_dir),
                                          map_location='cpu')
                else:
                    ckpt_path = os.path.join(input_dir, sub_dir,
                                             'state_dict.pth')
                    weight = torch.load(ckpt_path, map_location='cpu')
                    model.load_state_dict(weight)
                    del weight

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices # noqa
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        optim_cfg.learning_rate = (optim_cfg.learning_rate *
                                   train_cfg.gradient_accumulation_steps *
                                   train_cfg.train_batch_size *
                                   accelerator.num_processes)

    generator_parameters_with_lr = {
        'params': generator.parameters(),
        'lr': optim_cfg.gen_learning_rate
    }
    generater_params_to_optimize = [generator_parameters_with_lr]

    discriminator_parameters_with_lr = {
        'params': generator.parameters(),
        'lr': optim_cfg.disc_learning_rate
    }
    discriminator_params_to_optimize = [discriminator_parameters_with_lr]

    if optim_cfg.use_8bit_adam and not args.optimizer.lower() == 'adamw':
        logger.warning('use_8bit_adam is ignored when optimizer is not set to '
                       "'AdamW'. Optimizer was set to "
                       f'{args.optimizer.lower()}')

    if optim_cfg.optimizer.lower() == 'adamw':
        if optim_cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError('To use 8-bit Adam, please install the '
                                  'bitsandbytes library: `pip install '
                                  'bitsandbytes`.')

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer_gen = optimizer_class(
            generater_params_to_optimize,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            weight_decay=optim_cfg.adam_weight_decay,
            eps=optim_cfg.adam_epsilon,
        )

        optimizer_disc = optimizer_class(
            discriminator_params_to_optimize,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            weight_decay=optim_cfg.adam_weight_decay,
            eps=optim_cfg.adam_epsilon,
        )

    elif optim_cfg.optimizer.lower() == 'prodigy':
        try:
            import prodigyopt
        except ImportError:
            raise ImportError('To use Prodigy, please install the prodigyopt '
                              'library: `pip install prodigyopt`')

        optimizer_class = prodigyopt.Prodigy

        if optim_cfg.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's "
                'generally better to set learning rate around 1.0')

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            beta3=optim_cfg.prodigy_beta3,
            weight_decay=optim_cfg.adam_weight_decay,
            eps=optim_cfg.adam_epsilon,
            decouple=optim_cfg.prodigy_decouple,
            use_bias_correction=optim_cfg.prodigy_use_bias_correction,
            safeguard_warmup=optim_cfg.prodigy_safeguard_warmup,
        )

    train_dataset = HalfDataset(cfg.img_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=train_cfg.train_batch_size,
        num_workers=train_cfg.dataloader_num_workers)

    val_dataset = HalfDataset(cfg.img_path, split_type='valid')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=1,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_cfg.gradient_accumulation_steps)
    if train_cfg.max_train_steps is None:
        train_cfg.max_train_steps = train_cfg.num_train_epochs * \
            num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    generator, discriminator, optimizer_gen, optimizer_disc, \
        train_dataloader, loss_gan, loss_l1, loss_hist, \
            loss_style = accelerator.prepare(
                generator, discriminator, optimizer_gen,
                optimizer_disc, train_dataloader,
                loss_gan, loss_l1, loss_hist, loss_style)

    # We need to recalculate our total training steps as the size
    # of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        train_cfg.max_train_steps = train_cfg.num_train_epochs * \
            num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_cfg.num_train_epochs = math.ceil(train_cfg.max_train_steps /
                                           num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our
    # configuration. The trackers initializes automatically on the
    # main process.
    if accelerator.is_main_process:
        # tracker_config = dict(vars(cfg))
        # tracker_config = OmegaConf.to_container(cfg, resolve=True)

        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, dict):
                    # Recursively flatten the dictionary
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, tuple) or isinstance(v, list):
                    items.append((new_key, torch.tensor(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        # flat_config = flatten_dict(tracker_config)
        OmegaConf.save(cfg, os.path.join(train_cfg.output_dir, 'config.yaml'))

        if args.report_to == 'wandb':
            init_kwargs = {
                'wandb': {
                    'dir': os.path.abspath(train_cfg.output_dir),
                    # 'config': tracker_config,
                    'name': timestamp,
                }
            }
        else:
            init_kwargs = {}
        accelerator.init_trackers(
            args.tracker_project_name,
            init_kwargs=init_kwargs,
            #   config=flat_config
        )

    # Train!
    total_batch_size = train_cfg.train_batch_size * accelerator.num_processes \
        * train_cfg.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num batches each epoch = {len(train_dataloader)}')
    logger.info(f'  Num Epochs = {train_cfg.num_train_epochs}')
    logger.info('  Instantaneous batch size per device = '
                f'{train_cfg.train_batch_size}')
    logger.info('  Total train batch size (w. parallel, distributed & '
                f'accumulation) = {total_batch_size}')
    logger.info('  Gradient Accumulation steps = '
                f'{train_cfg.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {train_cfg.max_train_steps}')
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if train_cfg.resume_from_checkpoint:
        ckpt_path = train_cfg.resume_from_checkpoint
        accelerator.print(f'Resuming from checkpoint {ckpt_path}')
        accelerator.load_state(ckpt_path)
        global_step = int(ckpt_path.split('-')[-1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, train_cfg.max_train_steps),
        initial=initial_global_step,
        desc='Steps',
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(first_epoch, train_cfg.num_train_epochs):
        for _, batch_data in enumerate(train_dataloader):
            real_A_128 = batch_data['A']
            real_B = batch_data['B']
            if torch.cuda.is_available():
                real_A_128 = real_A_128.cuda()
                real_B = real_B.cuda()

            # synthesizing fake images
            fake_B_512 = generator(real_A_128).cuda()
            fake_B = CenterCrop(256)(fake_B_512).cuda()
            real_A = RandomResizedCrop(256)(real_A_128)
            # training on discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            disc_fake = discriminator(fake_AB.detach())
            loss_D_fake = loss_gan(disc_fake, target_is_real=False)

            real_AB = torch.cat((real_A, real_B), 1)
            disc_real = discriminator(real_AB)
            loss_D_real = loss_gan(disc_real, target_is_real=True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            optimizer_disc.zero_grad()
            loss_D.backward()
            optimizer_disc.step()

            # training on generator
            fake_B_B256 = torch.cat((real_B, fake_B), 1)
            disc_fake = discriminator(fake_B_B256)

            loss_GAN = loss_gan(disc_fake, True)
            loss_L1 = train_cfg.lambda_A * loss_l1(fake_B, real_B)
            style_loss = train_cfg.lambda_B * loss_style(fake_B, real_B)

            hist_loss = loss_hist(fake_B, real_B)

            loss_G = loss_L1 + style_loss + loss_GAN + hist_loss

            optimizer_gen.zero_grad()
            loss_G.backward()
            optimizer_gen.step()

            # loss = loss.mean()

            # log_loss += loss.detach()
            # accelerator.backward(loss)
            # if accelerator.sync_gradients:
            #     params_to_clip = controlnet.parameters()
            #     accelerator.clip_grad_norm_(params_to_clip,
            #                                 train_cfg.max_grad_norm)
            #     if getattr(train_cfg, 'train_transformer', False):
            #         accelerator.clip_grad_norm_(transformer.parameters(),
            #                                     train_cfg.max_grad_norm)
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization
            # step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % train_cfg.checkpointing_steps == 0:
                        save_path = os.path.join(train_cfg.output_dir,
                                                 f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f'Saved state to {save_path}')

                        # check if this save would
                        # set us over the `checkpoints_total_limit`
                        remove_old_checkpoints(
                            train_cfg.checkpoints_total_limit, train_cfg,
                            logger)

                    if global_step % train_cfg.validation_steps == 0:
                        if train_cfg.num_validation_images > 0:
                            log_validation(vae,
                                           transformer,
                                           controlnet,
                                           context_encoder,
                                           val_dataloader,
                                           args,
                                           cfg,
                                           accelerator,
                                           global_step,
                                           logger,
                                           weight_dtype=weight_dtype)

                # log_loss = accelerator.gather(
                #     log_loss / accelerator.gradient_accumulation_steps)
                # log_loss = log_loss.mean().item()
                logs = {'loss_G': loss_G.mean().item()}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= train_cfg.max_train_steps:
                break

    accelerator.end_training()


if __name__ == '__main__':
    args = parse_args()
    main(args)
