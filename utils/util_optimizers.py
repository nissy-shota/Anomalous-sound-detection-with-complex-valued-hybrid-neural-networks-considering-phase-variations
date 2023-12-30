from omegaconf import DictConfig
from torch import optim


def get_optimizer_scheduler(model, config: DictConfig):
    """
    Instantiate optimizer.
    """

    if config.optimizer.name == "Adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            weight_decay=config.optimizer.weight_decay,
            lr=config.optimizer.learning_rate,
        )
    elif config.optimizer.name == "RAdam":
        optimizer = optim.RAdam(
            params=model.parameters(),
            weight_decay=config.optimizer.weight_decay,
            lr=config.optimizer.learning_rate,
        )
    elif config.optimizer.name == "SGD":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError("Input Adam, SGD or RAdam")

    # optional
    if config.scheduler.name == "StepLR":

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.lr_step_size,
            gamma=config.scheduler.lr_gamma,
        )
    elif config.scheduler.name == "CosineAnnealingLR":

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.eta_min,
        )
    elif config.scheduler.name == "CosineLRScheduler":
        from timm.scheduler import CosineLRScheduler

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config.training.epochs,
            lr_min=config.scheduler.lr_min,
            warmup_t=config.scheduler.warmup_t,
            warmup_lr_init=config.scheduler.warmup_lr_init,
            warmup_prefix=config.scheduler.warmup_prefix,
        )

    return optimizer, scheduler
