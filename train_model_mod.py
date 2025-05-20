import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# First-party
from neural_lam import constants, utils
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.weather_dataset import WeatherDataset

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
}


def main():
    """
    Main function for training and evaluating models with dynamic LR, early stopping,
    mixed precision, gradient accumulation, profiling and DDP.
    """
    parser = ArgumentParser(description="Train or evaluate NeurWP models for LAM")
    # 新增 GPU 参数
    parser.add_argument("--gpus", type=int, default=2,
                        help="Number of GPUs to use (default: all available)")
    # General options
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_efm",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 4)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=2,
        help="Number of on-mesh GNN layers in encoder GNN (default: 2)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=2,
        help="Number of on-mesh GNN layers in prior GNN (default: 2)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--learn_prior",
        type=int,
        default=1,
        help="If the prior should be learned as a mapping from previous state "
        "and forcing, otherwise static with mean 0 (default: 1 (yes))",
    )
    parser.add_argument(
        "--vertical_propnets",
        type=int,
        default=0,
        help="If PropagationNets should be used for all vertical message "
        "passing (g2m, m2g, up in hierarchy), in deterministic models."
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        type=int,
        default=0,
        help="Train only on control member of ensemble data "
        "(default: 0 (False))",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--sample_obs_noise",
        type=int,
        default=0,
        help="If observation noise should be sampled during rollouts (both "
        "training and eval), or just mean prediction used "
        "(default: 0 (no))",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    args = parser.parse_args()

    # 参数检查
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (None, "val", "test"), f"Unknown eval setting: {args.eval}"

    # 随机 run id
    random_run_id = random.randint(0, 9999)
    
    # --- GPU 设备配置 ---
    if args.gpus is None:
        gpu_count = torch.cuda.device_count()  
    else:
        gpu_count = args.gpus  
    accelerator = "gpu" if gpu_count > 0 else "cpu"

    # 固定随机种子
    seed.seed_everything(args.seed)

    # 数据加载：增加 pin_memory=True, num_workers 提升
    train_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            args.dataset,
            pred_length=args.ar_steps,
            split="train",
            subsample_step=args.step_length,
            subset=bool(args.subset_ds),
            control_only=args.control_only,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers * 2,
        pin_memory=True,
        prefetch_factor=2,
    )
    max_pred_length = (65 // args.step_length) - 2
    val_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            args.dataset,
            pred_length=max_pred_length,
            split="val",
            subsample_step=args.step_length,
            subset=bool(args.subset_ds),
            control_only=args.control_only,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers * 2,
        pin_memory=True,
        prefetch_factor=2,
    )

    # 加载或初始化模型
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)
    # Ensure lr exists in hparams for old checkpoints
    if not hasattr(model.hparams, 'lr'):
        model.hparams.lr = args.lr

    # 训练名称
    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix += f"eval-{args.eval}-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # 回调：保存检查点 + 早停 + lr 监控
    checkpoint_callbacks = []
    checkpoint_callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        )
    )
    for unroll_time in constants.VAL_STEP_CHECKPOINTS:
        metric_name = f"val_loss_unroll{unroll_time}"
        checkpoint_callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename=f"min_{metric_name}",
                monitor=metric_name,
                mode="min",
            )
        )
    early_stop = EarlyStopping(monitor="val_mean_loss", patience=10, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 分布式策略
    strategy = "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"

    # Trainer 初始化
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        accelerator=accelerator,
        devices=gpu_count,
        strategy=strategy,
        precision=16,                    # 混合精度
        accumulate_grad_batches=4,      # 梯度累积
        callbacks=[*checkpoint_callbacks, early_stop, lr_monitor],
        check_val_every_n_epoch=args.val_interval,
        log_every_n_steps=1,
        profiler="pytorch",           # 性能分析
    )

    if trainer.global_rank == 0:
        try:
            utils.init_wandb_metrics(trainer.logger)
        except AttributeError:
            # Newer Lightning/WandB version may not support define_metric
            print("Warning: init_wandb_metrics skipped (define_metric not available)")

    # 评估 or 训练
    if args.eval:
        split = args.eval
        eval_loader = val_loader if split == "val" else torch.utils.data.DataLoader(
            WeatherDataset(
                args.dataset,
                pred_length=max_pred_length,
                split="test",
                subsample_step=args.step_length,
                subset=bool(args.subset_ds),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers * 2,
            pin_memory=True,
            prefetch_factor=2,
        )
        print(f"Running evaluation on {split}")
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # 可选：LR Finder
        from pytorch_lightning.tuner import Tuner
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        print(f"Setting learning rate to {new_lr}")

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


if __name__ == "__main__":
    main()


# 注意事项：
# 每个 LightningModule (GraphCast/GraphFM/GraphEFM) 中需实现 configure_optimizers：
#
# def configure_optimizers(self):
#     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#     scheduler = {
#         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="min",
#             factor=0.5,
#             patience=5,
#             verbose=True
#         ),
#         "monitor": "val_mean_loss",
#         "interval": "epoch",
#         "frequency": 1,
#     }
#     return {"optimizer": optimizer, "lr_scheduler": scheduler}
