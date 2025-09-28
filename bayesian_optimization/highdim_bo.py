import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import time
import torch
import tqdm
import yaml
from argparse import ArgumentParser
from attrdict import AttrDict
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bayeso_benchmarks.inf_dim_ackley import Ackley
from bayeso_benchmarks.inf_dim_cosines import Cosines
from bayeso_benchmarks.inf_dim_rastrigin import Rastrigin
from bayeso_benchmarks.two_dim_dropwave import DropWave
from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice
from bayeso_benchmarks.two_dim_michalewicz import TranslatedMichalewicz
from bayeso_benchmarks.three_dim_hartmann3d import Hartmann3D

from utils.acquisition import UCB, EI
from utils.misc import load_module

from utils.paths import results_path

def main():
    parser = ArgumentParser()

    parser.add_argument('--mode', choices=['bo', 'plot'], default='bo')
    parser.add_argument('--time_comparison', action='store_true', default=False)

    parser.add_argument('--objective',
                        choices=['ackley',
                                 'cosine',
                                 'rastrigin',
                                 'rosenbrock',
                                 'shubert',
                                 'xinsheyang',
                                 'gramacyandlee',
                                 'dropwave',
                                 'goldsteinprice',
                                 'michalewicz',
                                 'hartmann'],
                        default='ackley')
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--acquisition', choices=['ucb', 'ei'], default='ucb')

    parser.add_argument('--model', default='tnpa')
    parser.add_argument('--train_num_bootstrap', type=int, default=10)
    parser.add_argument('--train_num_steps', type=int, default=100000)
    parser.add_argument('--train_max_num_points', type=int, default=128)
    parser.add_argument('--train_min_num_points', type=int, default=30)

    parser.add_argument('--num_task', type=int, default=100)
    parser.add_argument('--num_iter', type=int, default=100)
    parser.add_argument('--num_initial_design', type=int, default=1)
    parser.add_argument('--num_bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temperature', type=int, default=0)
    parser.add_argument('--train_seed', type=int, default=0) 
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--device', type=str, default="0")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.mode == 'bo':
        if args.model == 'gp':
            gp(
                expid = args.expid,
                obj_func=args.objective,
                dim_problem=args.dimension,
                result_path=results_path,
                acq_func=args.acquisition,
                num_task=args.num_task,
                num_iter=args.num_iter,
                num_initial_design=args.num_initial_design,
                seed=args.seed
            )
        else:
            bo(
                expid = args.expid,
                train_seed = args.train_seed,
                temperature = args.temperature,
                obj_func=args.objective,
                dim_problem=args.dimension,
                result_path=results_path,
                acq_func=args.acquisition,
                model_name=args.model,
                train_num_bootstrap=args.train_num_bootstrap,
                train_num_step=args.train_num_steps,
                train_max_num_points=args.train_max_num_points,
                train_min_num_points=args.train_min_num_points,
                num_task=args.num_task,
                num_iter=args.num_iter,
                num_initial_design=args.num_initial_design,
                num_bootstrap=args.num_bootstrap,
                seed=args.seed
            )

    elif args.mode == 'plot':
        plot(
            train_seed = args.train_seed,
            temperature = args.temperature,
            dimension=args.dimension,
            result_path=results_path,
            num_iter=args.num_iter,
            seed=args.seed
        )

    else:
        raise NotImplementedError

def gp(
        expid: str,
        obj_func: str,
        dim_problem: int,
        result_path: str,
        acq_func: str = 'ei',
        num_task: int = 100,
        num_iter: int = 50,
        num_initial_design: int = 1,
        seed: int = 42
):
    assert isinstance(dim_problem, int) and (dim_problem > 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = osp.join(result_path, 'highdim_bo', f'{dim_problem}D', f'{obj_func}', 'gp', 'matern_gamma')
    if not osp.isdir(root):
        os.makedirs(root)

    print(f"BO Experiment: [{dim_problem}D {obj_func}] [{acq_func}] {'-'.join(root.split('/')[-2:])}")
    print(f"Device: {device}\n")

    if obj_func == 'ackley':
        obj = Ackley(dim_problem=dim_problem)
    elif obj_func == 'cosine':
        obj = Cosines(dim_problem=dim_problem)
    elif obj_func == 'rastrigin':
        obj = Rastrigin(dim_problem=dim_problem)

    elif dim_problem == 2:
        if obj_func == 'dropwave':
            obj = DropWave()
        elif obj_func == 'goldsteinprice':
            obj = GoldsteinPrice()
        elif obj_func == 'michalewicz':
            obj = TranslatedMichalewicz()
        else:
            raise NotImplementedError

    elif dim_problem == 3:
        if obj_func == 'hartmann':
            obj = Hartmann3D()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    if dim_problem == 1:
        lb, ub = obj.get_bounds()[0]
        lb, ub = max(-2, lb), min(2, ub)
        bound = torch.tensor([[lb], [ub]], dtype=torch.float, device=device)
    else:
        lb, ub = obj.get_bounds().transpose()
        lb, ub = np.where(lb < -2, -2, lb), np.where(ub > 2, 2, ub)
        bound = torch.tensor([lb, ub], dtype=torch.float, device=device)

    regrets = np.zeros((num_task, num_iter + 1))
    times = np.zeros((num_task, num_iter + 1))
    for i in tqdm.tqdm(range(1, num_task + 1), unit='task', ascii=True):
        seed_ = seed * i

        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)
        np.random.seed(seed_)

        global_min = obj.global_minimum
        # initial design & to tensor
        x = lb + (ub - lb) * np.random.rand(num_initial_design, dim_problem)
        y = obj.output(x)
        x = torch.tensor(x, dtype=torch.float, device=device)
        y = torch.tensor(y, dtype=torch.float, device=device)
        min_values = [y.min().cpu().numpy().item()]

        time_list = [0]
        start_time = time.time()

        gp_model = SingleTaskGP(x, y).to(device)
        mll = ExactMarginalLogLikelihood(
            likelihood=gp_model.likelihood,
            model=gp_model
        ).to(device)

        raw_samples = 100
        for _ in range(num_iter):
            try:
                gp_model.train()
                mll.train()
                fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={'disp': False})
                gp_model.eval()
                mll.eval()

                if acq_func == 'ucb':
                    acq = UpperConfidenceBound(
                        model=gp_model,
                        beta=0.1,
                        maximize=False
                    )
                elif acq_func == 'ei':
                    acq = ExpectedImprovement(
                        model=gp_model,
                        best_f=min_values[-1],
                        maximize=False
                    )
                else:
                    raise NotImplementedError

                new_point, acq_value = optimize_acqf(
                    acq_function=acq,
                    bounds=bound,
                    q=1,
                    num_restarts=50,
                    raw_samples=raw_samples,
                    options={},
                )

                x_new = new_point.to(device)
                y_new = torch.tensor(obj.output(new_point.cpu().numpy()),
                                     dtype=torch.float, device=device)

                x = torch.cat([x, x_new], dim=-2)
                y = torch.cat([y, y_new], dim=-2)

                old_model_state = gp_model.state_dict()
                gp_model = SingleTaskGP(x, y).to(device)
                mll = ExactMarginalLogLikelihood(
                    likelihood=gp_model.likelihood,
                    model=gp_model
                ).to(device)
                gp_model.load_state_dict(old_model_state)

                current_min = y.min()
                min_values.append(current_min.cpu().numpy().item())
                time_list.append(time.time() - start_time)

            except RuntimeError:
                break

        if len(min_values) == num_iter + 1:
            regrets[i - 1] = np.array(min_values) - global_min
            times[i - 1] = np.array(time_list)

    exp_results = {'regrets': regrets, 'times': times}
    np.save(osp.join(root, f'results_{acq_func}_{seed}.npy'), exp_results)


def bo(
        expid: str,
        train_seed: int,
        temperature: int,
        obj_func: str,
        dim_problem: int,
        result_path: str,
        acq_func: str = 'ei',
        model_name: str = 'tnpa',
        train_num_bootstrap: int = 10,
        train_num_step: int = 100000,
        train_max_num_points: int = 128,
        train_min_num_points: int = 30,
        num_task: int = 100,
        num_iter: int = 50,
        num_initial_design: int = 1,
        num_bootstrap: int = 200,
        seed: int = 42
):
    assert isinstance(dim_problem, int) and (dim_problem > 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f'configs/gp/{model_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config['dim_x'] = dim_problem

    model_cls = getattr(load_module(f"models/{model_name}.py"), model_name.upper())
    model = model_cls(**config).to(device)

    # expid = f'min{train_min_num_points}_max{train_max_num_points}_{int(train_num_step / 10000)}'
    expid = f't{temperature}_s{train_seed}'

    root = osp.join(result_path, 'highdim_bo', f'{dim_problem}D', f'{obj_func}', model_name, expid)
    if not osp.isdir(root):
        os.makedirs(root)

    if dim_problem == 1:
        ckpt_path = osp.join(results_path, '1d', model_name, expid, 'ckpt.tar')
    else:
        ckpt_path = osp.join(results_path, 'highdim_gp', f'{dim_problem}D', model_name, expid, 'ckpt.tar')

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.model)

    print(f"Model checkpoint: {ckpt_path}")
    print(f"BO Experiment: [{dim_problem}D {obj_func}] [{acq_func}] {'-'.join(root.split('/')[-2:])}")
    print(f"Device: {device}\n")

    if obj_func == 'ackley':
        obj = Ackley(dim_problem=dim_problem)
    elif obj_func == 'cosine':
        obj = Cosines(dim_problem=dim_problem)
    elif obj_func == 'rastrigin':
        obj = Rastrigin(dim_problem=dim_problem)

    elif dim_problem == 2:
        if obj_func == 'dropwave':
            obj = DropWave()
        elif obj_func == 'goldsteinprice':
            obj = GoldsteinPrice()
        elif obj_func == 'michalewicz':
            obj = TranslatedMichalewicz()
        else:
            raise NotImplementedError

    elif dim_problem == 3:
        if obj_func == 'hartmann':
            obj = Hartmann3D()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    if dim_problem == 1:
        lb, ub = obj.get_bounds()[0]
        lb, ub = max(-2, lb), min(2, ub)
        bound = torch.tensor([[lb], [ub]], dtype=torch.float, device=device)
    else:
        lb, ub = obj.get_bounds().transpose()
        lb, ub = np.where(lb < -2, -2, lb), np.where(ub > 2, 2, ub)
        bound = torch.tensor([lb, ub], dtype=torch.float, device=device)

    regrets = np.zeros((num_task, num_iter + 1))
    times = np.zeros((num_task, num_iter + 1))
    for i in tqdm.tqdm(range(1, num_task + 1), unit='task', ascii=True):
        seed_ = seed * i

        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)
        np.random.seed(seed_)

        global_min = obj.global_minimum
        # initial design & to tensor
        x = lb + (ub - lb) * np.random.rand(num_initial_design, dim_problem)
        y = obj.output(x)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float, device=device).unsqueeze(0)

        batch = AttrDict()
        batch.xc = x
        batch.yc = y
        min_values = [batch.yc.min().cpu().numpy().item()]

        if acq_func == 'ucb':
            acq = UCB(
                model=model,
                observations=batch,
                beta=0.1,
                num_bs=num_bootstrap,
                maximize=False
            )
        elif acq_func == 'ei':
            acq = EI(
                model=model,
                observations=batch,
                best_f=min_values[-1],
                num_bs=num_bootstrap,
                maximize=False
            )
        else:
            raise NotImplementedError

        time_list = [0]
        start_time = time.time()
        model.eval()

        raw_samples = 100
        for _ in range(num_iter):
            new_point, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bound,
                q=1,
                num_restarts=50,
                raw_samples=raw_samples,
                options={},
            )

            x_new = new_point.unsqueeze(0).to(device)
            y_new = torch.tensor(obj.output(new_point.cpu().numpy()),
                                 dtype=torch.float, device=device).unsqueeze(0)

            batch.xc = torch.cat([batch.xc, x_new], dim=-2)
            batch.yc = torch.cat([batch.yc, y_new], dim=-2)
            current_min = batch.yc.min()
            min_values.append(current_min.cpu().numpy().item())

            acq.obs = batch
            if acq_func == 'ei':
                acq.best_f = torch.tensor(min_values[-1])
            time_list.append(time.time() - start_time)

        regrets[i - 1] = np.array(min_values) - global_min
        times[i - 1] = np.array(time_list)

    exp_results = {'regrets': regrets, 'times': times}
    np.save(osp.join(root, f'results_{acq_func}_{seed}.npy'), exp_results)

def plot(
        train_seed: int,
        temperature: int,
        dimension: int,
        result_path: str,
        num_iter: int = 50,
        seed: int = 42
):
    color = {
        'np': 'navy',
        'anp': 'darkgreen',
        'bnp': 'darkgoldenrod',
        'cnp': 'darkred',
        'tnpa': 'deepskyblue',
    }
    root = osp.join(result_path, 'highdim_bo')
    model_names = ['NP', 'ANP', 'BNP', 'CNP', 'TNP']
    table = {2: {'function': ['Ackley', 'Dropwave', 'Michalewicz'], 'min': 30, 'max': 128, 'step': 100000},
             3: {'function': ['Ackley', 'Cosine', 'Rastrigin'], 'min': 64, 'max': 256, 'step': 100000}}

    acq_func = 'ucb'
    
    if dimension not in [2, 3]:
        raise ValueError("Dimension must be either 2 or 3")
    
    functions = table[dimension]['function']

    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    
    for k, m in enumerate(color.keys()):
        for j, function in enumerate(functions):
            func = function.lower()

            
            expid = 'to'
            pth = osp.join(root, f'{dimension}D', f'{func}', m, expid, f'results_{acq_func}_{seed}.npy')
            if osp.isfile(pth):
                exp_results = np.load(pth, allow_pickle=True).item()
                regrets = exp_results['regrets']
                mean, std = regrets.mean(0), regrets.std(0) * 0.2
                axes[k][j].plot(np.arange(num_iter + 1), mean, label=f'{model_names[k]}-V', lw=2.1, color=color[m], linestyle='-')
                axes[k][j].fill_between(np.arange(num_iter + 1), mean - std, mean + std, alpha=0.1, color=color[m])

            
            expid_Gen = 'tn'
            pth_Gen = osp.join(root, f'{dimension}D', f'{func}', m, expid_Gen, f'results_{acq_func}_{seed}.npy')
            if osp.isfile(pth_Gen):
                exp_results_Gen = np.load(pth_Gen, allow_pickle=True).item()
                regrets_Gen = exp_results_Gen['regrets']
                mean_Gen, std_Gen = regrets_Gen.mean(0), regrets_Gen.std(0) * 0.2
                axes[k][j].plot(np.arange(num_iter + 1), mean_Gen, label=f'{model_names[k]}-Gen', lw=2.1, color=color[m], linestyle='--')
                axes[k][j].fill_between(np.arange(num_iter + 1), mean_Gen - std_Gen, mean_Gen + std_Gen, alpha=0.1, color=color[m])

            axes[k][j].set_title(f'{model_names[k]} - {function} ({dimension}D)', fontsize=14)
            axes[k][j].legend(loc='upper right', fontsize=10)
            axes[k][j].grid(ls=':')

            if k == 4:
                axes[k][j].set_xlabel('Iterations', fontsize=12)
            if j == 0:
                axes[k][j].set_ylabel('Regret', fontsize=12)
    
    plt.tight_layout()
    figname = f'{dimension}D_BO_{acq_func}.png'
    plt.savefig(osp.join(root, figname), dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
