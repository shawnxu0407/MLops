import os
import sys


## Define the evvironment 
dic_path='D:/RL_Finance/MLops/fslab/lab04'

sys.path.append(dic_path)
os.chdir(dic_path)


import argparse
from pathlib import Path
from text_recognizer import callbacks as cb
import importlib
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from text_recognizer import lit_models
from training.util import DATA_CLASS_MODULE, import_class, MODEL_CLASS_MODULE, setup_data_and_model_from_args
from text_recognizer.data.iam_lines import IAMLines, PreloadedIAMLines
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only



sys.argv = [arg for arg in sys.argv if not arg.startswith("--f=")]

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments manually
    parser.add_argument('--max_epochs', type=int, default=1, help='Max number of epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, help='Precision of the model')
    parser.add_argument('--limit_train_batches', type=float, default=0.1, help='Limit number of training batches')
    parser.add_argument('--limit_test_batches', type=float, default=0.1, help='Limit number of test batches')
    parser.add_argument('--limit_val_batches', type=float, default=0.1, help='Limit number of validation batches')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='Logging frequency in steps')
    parser.add_argument('--wandb', action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use')


    # Basic arguments
    parser.add_argument(
        "--check_val_every_n_epoch", 
        type=int, 
        default=1, 
        help="Number of epochs between validation checks"
    )
    parser.add_argument(
        "--data_class",
        type=str,
        default="IAMLines",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="LineCNNTransformer",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu). Defaults to cuda if available."
    )


    
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()

    temp_args, unknown = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)




def main():
    # Remove Jupyter's --f=... argument
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--f=")]

    parser = _setup_parser()
    args = parser.parse_args()
    ## change the model type to be tranformer


    ## import the model
    ## import the model

    _ , model = setup_data_and_model_from_args(args)
    lit_model_class = lit_models.TransformerLitModel
    lit_model = lit_model_class(args=args, model=model)
    lit_model=lit_model.to(args.device)

    ## load the train/vali/test data loader
    data_module = PreloadedIAMLines(args)  
    # Setup data (loads everything into RAM)
    data_module.setup()

    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    callbacks = []


    if args.wandb:
        logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))


    if args.wandb and args.loss in ("transformer",):
        callbacks.append(cb.ImageToTextLogger())

    trainer = pl.Trainer(
                max_epochs=args.max_epochs,
                precision=args.precision,
                limit_train_batches=args.limit_train_batches,
                limit_test_batches=args.limit_test_batches,
                limit_val_batches=args.limit_val_batches,
                logger=logger,
                check_val_every_n_epoch=args.check_val_every_n_epoch,
                enable_progress_bar=False,
                callbacks=callbacks,
                )

    ## trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(lit_model, datamodule=data_module)


if __name__ == "__main__":
    main()
