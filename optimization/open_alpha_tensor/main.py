import json
import os
from argparse import ArgumentParser
from pathlib import Path

from open_alpha_tensor import train_alpha_tensor


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
    parser.add_argument("--checkpoint_data_dir", type=str, default=None)
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
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("extra_devices", nargs="*", type=str, default=[])
    parser.set_defaults(**config)
    args = parser.parse_args()

    cardinality_vector = args.cardinality_vector
    N_bar = args.n_bar
    input_size = args.matrix_size**2
    n_steps = _compute_largest_divisor(input_size)
    n_actions = cardinality_vector ** (3 * input_size // n_steps)
    loss_params = (args.alpha, args.beta)

    train_alpha_tensor(
        tensor_length=args.action_memory + 1,
        input_size=input_size,
        scalars_size=1,
        emb_dim=args.embed_dim,
        n_steps=n_steps,
        n_logits=n_actions,
        n_samples=args.actions_sampled,
        device=args.device,
        len_data=args.len_data,
        n_synth_data=args.n_synth_data,
        pct_synth=args.pct_synth,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_steps=args.lr_decay_steps,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        loss_params=loss_params,
        limit_rank=args.limit_rank,
        random_seed=args.random_seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_data_dir=args.checkpoint_data_dir,
        n_actors=args.n_actors,
        mc_n_sim=args.mc_n_sim,
        n_cob=args.n_cob,
        cob_prob=args.cob_prob,
        data_augmentation=args.data_augmentation or False,
        N_bar=N_bar,
        extra_devices=args.extra_devices,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
