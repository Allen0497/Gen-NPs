import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda') # 'cpu' to use cpu
    parser.add_argument('--device_num', type=str, default="0")
    # wheel
    parser.add_argument("--cmab_data", choices=["wheel"], default="wheel")
    parser.add_argument("--cmab_wheel_delta", type=float, default=0.5)
    parser.add_argument("--cmab_mode", choices=["train", "eval", "plot", "evalplot"], default="train")
    parser.add_argument('--cmab_num_bs', type=int, default=10)
    parser.add_argument("--cmab_train_update_freq", type=int, default=1)
    parser.add_argument("--cmab_train_num_batches", type=int, default=1)
    parser.add_argument("--cmab_train_batch_size", type=int, default=8)
    parser.add_argument("--cmab_train_seed", type=int, default=0)
    parser.add_argument("--cmab_train_reward", type=str, default="all")
    parser.add_argument("--cmab_eval_method", type=str, default="ucb")
    parser.add_argument("--cmab_eval_num_contexts", type=int, default=2000)
    parser.add_argument("--cmab_eval_seed_start", type=int, default=0)
    parser.add_argument("--cmab_eval_seed_end", type=int, default=49)
    parser.add_argument("--cmab_plot_seed_start", type=int, default=0)
    parser.add_argument("--cmab_plot_seed_end", type=int, default=49)

    parser.add_argument('--use_dsr', type=int, default=1, help='Enable DSR regularization')
    parser.add_argument('--lambda1', type=float, default=0.05, help='Coefficient for Hessian trace term')
    parser.add_argument('--lambda2', type=float, default=0.005, help='Coefficient for Hessian Frobenius norm term')
    parser.add_argument('--hessian_samples', type=int, default=5, help='Number of samples for Hessian estimation')

    # Model
    parser.add_argument('--model', type=str, default="tnpa")

    # Training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--temperature', type=int, default=10000)
    parser.add_argument('--train_mode', choices=["v", "s"], default="s")
    
    args = parser.parse_args()

    return args