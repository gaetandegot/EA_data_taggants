from timm.optim import create_optimizer


def build_optimizer(model, args):

    optimizer = create_optimizer(args, model)

    return optimizer
