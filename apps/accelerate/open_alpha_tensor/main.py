import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch

try:
    from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
    from open_alpha_tensor.core.training import Trainer
except ImportError or ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
    from open_alpha_tensor.core.training import Trainer


def _compute_largest_divisor(n: int) -> int:
    """Compute the largest divisor of n."""
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i
    return 1


def main():
    config_file = Path(os.getenv("CONFIG_FILE", "config.json"))
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--action_memory", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--half", action="store_true")
    parser.add_argument("--len_data", type=int, default=100)
    parser.add_argument("--pct_synth", type=float, default=0.5)
    parser.add_argument("--n_synth_data", type=int, default=100)
    parser.add_argument("--limit_rank", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--matrix_size", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--actions_sampled", type=int, default=10)
    parser.add_argument("--n_actors", type=int, default=1)
    parser.add_argument("--mc_n_sim", type=int, default=100)
    parser.add_argument("--n_cob", type=int, default=100000)
    parser.add_argument("--cob_prob", type=float, default=0.9983)  # 1 - 0.0017
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--cardinality_vector", type=int, default=5)
    parser.add_argument(
        "--n_bar",
        type=int,
        default=100,
        help="N_bar parameter for policy temperature.",
    )
    parser.set_defaults(**config)
    args = parser.parse_args()

    cardinality_vector = args.cardinality_vector
    N_bar = args.n_bar
    input_size = args.matrix_size**2
    n_steps = _compute_largest_divisor(input_size)
    n_actions = cardinality_vector ** (3 * input_size // n_steps)
    print(n_actions)
    loss_params = (args.alpha, args.beta)
    print("Creating model...")
    model = AlphaTensorModel(
        tensor_length=args.action_memory + 1,
        input_size=input_size,
        scalars_size=1,
        emb_dim=args.embed_dim,
        n_steps=n_steps,
        n_logits=n_actions,
        n_samples=args.actions_sampled,
    )
    print("Model created.")
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    games_store_dir = Path("games")

    if (
        args.checkpoint_dir is not None
        and Path(args.checkpoint_dir).exists()
        and len(list(Path(args.checkpoint_dir).glob("*.pt"))) > 0
    ):

        def key_func(x):
            return int(x.stem.split("_")[-1])

        checkpoint_path = sorted(
            Path(args.checkpoint_dir).glob("*.pt"), key=key_func
        )[-1]
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = int(checkpoint_path.stem.split("_")[-1])
    else:
        last_epoch = 0

    trainer = Trainer(
        model=model,
        tensor_size=input_size,
        n_steps=n_steps,
        batch_size=args.batch_size,
        optimizer=optimizer,
        device=args.device,
        len_data=args.len_data,
        pct_synth=args.pct_synth,
        n_synth_data=args.n_synth_data,
        limit_rank=args.limit_rank,
        loss_params=loss_params,
        random_seed=args.random_seed,
        checkpoint_dir=args.checkpoint_dir,
        data_augmentation=args.data_augmentation or False,
        cob_prob=args.cob_prob,
        n_cob=args.n_cob,
    )

    # if games_store_dir contains games, load them
    if (
        games_store_dir.exists()
        and (games_store_dir / "game_data.json").exists()
    ):
        trainer.dataset.load_games(games_store_dir)

    # train for max_epochs
    trainer.train(
        n_epochs=args.max_epochs,
        n_games=args.n_actors,
        mc_n_sim=args.mc_n_sim,
        N_bar=N_bar,
        starting_epoch=last_epoch,
        initial_lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_steps=args.lr_decay_steps,
    )

    # save model and parameters
    torch.save(model.state_dict(), "final_model.pt")
    model_params = {
        "input_size": args.matrix_size**2,
        "tensor_length": args.action_memory + 1,
        "scalars_size": 1,
        "emb_dim": args.embed_dim,
        "n_steps": 1,
        "n_logits": n_actions,
        "n_samples": args.actions_sampled,
    }
    # save parameters in a json file
    with open("model_params.json", "w") as f:
        json.dump(model_params, f)


if __name__ == "__main__":
    main()
