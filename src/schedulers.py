from timm.scheduler import create_scheduler


def build_scheduler(optimizer, args):

    scheduler, _ = create_scheduler(args, optimizer)

    return scheduler
