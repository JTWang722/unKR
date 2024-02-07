# -*- coding: utf-8 -*-c
# from torch._C import T
# from train import Trainer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *


class DecayCallback(pl.Callback):
    def __init__(self, decay_epochs):
        super().__init__()
        self.decay_epochs = decay_epochs

    def on_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch in self.decay_epochs:
            pl_module.model.adjust_parameters(current_epoch)

def main(arg_path):
    print('This demo is for testing FocusE')
    args = setup_parser()  # 设置参数
    args = load_config(args, arg_path)
    seed_everything(args.seed)
    print(args.dataset_name)

    """set up sampler to datapreprocess"""  # 设置数据处理的采样过程
    train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的
    # print(train_sampler)
    test_sampler_class = import_class(f"unKR.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的

    """set up datamodule"""  # 设置数据模块
    data_class = import_class(f"unKR.data.{args.data_class}")  # 定义数据类 DataClass
    kgdata = data_class(args, train_sampler, test_sampler)

    """set up model"""
    model_class = import_class(f"unKR.model.{args.model_name}")
    model = model_class(args)

    """set up lit_model"""
    litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, args)

    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="unKR")
        logger.log_hyperparams(vars(args))

    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval_wmrr",
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )

    """ β衰减 """
    # decay_epochs = [100, 200, 300, 400, 500, 600, 700, 800]  # BaseModel=TransE
    # # decay_epochs = [25, 50, 75, 100, 125, 150, 175, 200]       # BaseModel=DistMult
    decay_callback = DecayCallback(range(args.max_epochs))

    """set up model save method"""
    # 目前是保存在验证集上MSE结果最好的模型
    # 模型保存的路径
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval_wmrr",
        mode="max",
        filename="{epoch}-{Eval_mrr:.5f}",
        dirpath=dirpath,
        save_weights_only=True,
        save_top_k=1,
    )
    callbacks = [early_callback, model_checkpoint, decay_callback]
    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus="0,",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    '''保存参数到config'''
    if args.save_config:
        save_config(args)

    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # 加载本次实验中dev上表现最好的模型，进行test
        path = model_checkpoint.best_model_path
    else:
        # path = args.checkpoint_dir
        path = "./output/confidence_prediction/ppi5k/FocusE/epoch=5-Eval_MSE=0.052.ckpt"
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)

if __name__ == "__main__":
    main(arg_path='../config/cn15k/FocusE_cn15k.yaml')
