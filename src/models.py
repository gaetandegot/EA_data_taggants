import torch

from timm.models import create_model


def build_deit(args):
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int(
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5
        )
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                        embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic',
            align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    return model


def build_resnet(args):
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        model.load_state_dict(checkpoint['model'], strict=False)

    return model


def build_convnext(args):
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(
                    prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            load(model, prefix=prefix)

            warn_missing_keys = []
            ignore_missing_keys = []
            for key in missing_keys:
                keep_flag = True
                for ignore_key in ignore_missing.split('|'):
                    if ignore_key in key:
                        keep_flag = False
                        break
                if keep_flag:
                    warn_missing_keys.append(key)
                else:
                    ignore_missing_keys.append(key)

            missing_keys = warn_missing_keys

            if len(missing_keys) > 0:
                print("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                print("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(ignore_missing_keys) > 0:
                print("Ignored weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, ignore_missing_keys))
            if len(error_msgs) > 0:
                print('\n'.join(error_msgs))

        load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    return model


def build_model(args, is_sign=False):

    if args.model.startswith('deit'):
        model = build_deit(args)
    elif args.model.startswith('resnet'):
        model = build_resnet(args)
    elif args.model.startswith('convnext'):
        model = build_convnext(args)
    elif args.model == 'vgg11':
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
        )
    elif args.model == 'densenet121':
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
        )
    elif args.model == 'mobilenetv2':
        model = create_model(
            'mobilenetv2_100',
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    model.to(args.device)

    if args.distributed and not is_sign:
        print(f"{args.gpu=}")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )

    return model
