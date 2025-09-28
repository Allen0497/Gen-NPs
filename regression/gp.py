import os
import os.path as osp
import argparse
import yaml
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from attrdict import AttrDict
from tqdm import tqdm
from copy import deepcopy

from data.gp import *
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def hessian_vector_product(loss, parameters, v):
    """
    Compute the Hessian-vector product Hv where H is the Hessian of the loss function
    and v is a vector, using Pearlmutter's algorithm.
    """
    grad_p = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
    flat_grad_p = torch.cat([g.view(-1) for g in grad_p])
    hvp = torch.autograd.grad(flat_grad_p, parameters, grad_outputs=v, retain_graph=True)
    return hvp

def estimate_hessian_trace(loss, parameters, num_samples=10):
    """
    Estimate the trace of the Hessian using Hutchinson's trace estimator.
    """
    parameters = list(parameters)
    num_params = sum(p.numel() for p in parameters)
    trace_estimate = 0.0
    
    for _ in range(num_samples):
        v = [torch.randn_like(p) for p in parameters]  # Random Rademacher vectors
        flat_v = torch.cat([vec.view(-1) for vec in v])
        norm_v = flat_v / flat_v.norm()
        v_normalized = []
        offset = 0
        for p in parameters:
            v_size = p.numel()
            v_normalized.append(norm_v[offset:offset+v_size].view(p.size()))
            offset += v_size
            
        hvp = hessian_vector_product(loss, parameters, v_normalized)
        trace_estimate += sum(torch.sum(vi * hvi) for vi, hvi in zip(v_normalized, hvp))
    
    return trace_estimate / num_samples

def estimate_hessian_frobenius_norm(loss, parameters, num_samples=5):
    """
    Estimate the Frobenius norm of the Hessian using a stochastic approximation.
    """
    parameters = list(parameters)
    frob_norm_estimate = 0.0
    
    for _ in range(num_samples):
        v = [torch.randn_like(p) for p in parameters]  # Random vectors
        flat_v = torch.cat([vec.view(-1) for vec in v])
        norm_v = flat_v / flat_v.norm()
        v_normalized = []
        offset = 0
        for p in parameters:
            v_size = p.numel()
            v_normalized.append(norm_v[offset:offset+v_size].view(p.size()))
            offset += v_size
            
        hvp = hessian_vector_product(loss, parameters, v_normalized)
        hvp_norm = torch.cat([h.view(-1) for h in hvp]).norm()
        frob_norm_estimate += hvp_norm ** 2
    
    return torch.sqrt(frob_norm_estimate / num_samples)

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'eval_all_metrics', 'plot'])
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default="0")

    # Data
    parser.add_argument('--max_num_points', type=int, default=50)

    # Model
    parser.add_argument('--model', type=str, default="tnpd")

    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--sgld', type=int, default=1)
    parser.add_argument('--temperature', type=int)

    #DSR regularization parameters
    parser.add_argument('--use_dsr', type=int, default=1, help='Enable DSR regularization')
    parser.add_argument('--lambda1', type=float, default=0.05, help='Coefficient for Hessian trace term')
    parser.add_argument('--lambda2', type=float, default=0.005, help='Coefficient for Hessian Frobenius norm term')
    parser.add_argument('--hessian_samples', type=int, default=5, help='Number of samples for Hessian estimation')

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # Plot
    parser.add_argument('--plot_seed', type=int, default=0)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=30)
    parser.add_argument('--plot_num_tar', type=int, default=10)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--init_kernel', type=str, default='rbf')
    parser.add_argument('--eval_kernel', type=str, default='rbf')
    parser.add_argument('--plot_kernel', type=str, default='rbf')
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.model == "cnp":
        args.temperature = 100000000
    elif args.model == "np":
        args.temperature = 10000000
    elif args.model == "anp":
        args.temperature = 10000000
    elif args.model == "canp":
        args.temperature = 100000000
    elif args.model == "bnp":
        args.temperature = 1000000
    elif args.model == "banp":
        args.temperature = 10000000
    elif args.model == "tnpa":
        args.temperature = 1000000
    elif args.model == "tnpd":
        args.temperature = 1000000000
    elif args.model == "tnpnd":
        args.temperature = 1000000000

    if args.expid is not None:
        args.root = osp.join(results_path, 'bound', 'gp', args.init_kernel, args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'gp', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd"]:
        model = model_cls(**config)
    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'eval_all_metrics':
        eval_all_metrics(args, model)
    elif args.mode == 'plot':
        plot(args, model)

def cal_grad_norm(grad):
    para_norm = 0
    for g in grad:
        para_norm += g.data.norm(2).item() ** 2
    return para_norm

def cal_grad_incohence(grad1, grad2):

    g_incohen = 0
    for (g1, g2) in zip(grad1, grad2): 
        g_incohen += (g1.data - g2.data).norm(2).item() ** 2
    return g_incohen

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)

    if args.init_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.init_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.init_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.init_kernel}')
    print(f"Initialing sampler with {args.init_kernel} kernel")

    sampler = GPSampler(kernel)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.model}-{args.expid}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    temperature = args.temperature
    g_incohen = []
    outer_g_inco_bound = 0
    dsr_values = []

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device='cuda')
        
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            outs = model(batch, num_samples=args.train_num_samples)
        else:
            outs = model(batch)

        if args.use_dsr:
            # Store original loss before adding regularization
            original_loss = outs.loss.item()
            
            # Estimate Hessian properties for DSR
            trace_estimate = estimate_hessian_trace(outs.loss, model.parameters(), num_samples=args.hessian_samples)
            frob_norm_estimate = estimate_hessian_frobenius_norm(outs.loss, model.parameters(), num_samples=args.hessian_samples)
            
            # Compute DSR regularization term
            dsr_term = args.lambda1 * trace_estimate + args.lambda2 * frob_norm_estimate
            dsr_values.append(dsr_term.item())
            
            # Add DSR term to the loss
            total_loss = outs.loss + dsr_term
            total_loss.backward()
            
            # Log the DSR components
            ravg.update('hessian_trace', trace_estimate.item())
            ravg.update('hessian_frob', frob_norm_estimate.item())
            ravg.update('dsr_term', dsr_term.item())
            ravg.update('original_loss', original_loss)
        else:
            outs.loss.backward()
        optimizer.step()
        scheduler.step()

        meta_loss_Str_K, meta_loss_S_K = 0, 0

        target_batch = AttrDict(xc=batch.xt, yc=batch.yt, xt=batch.xt, yt=batch.yt, x=batch.xt, y=batch.yt)
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            target_outs = model(target_batch, num_samples=args.train_num_samples)
        else:
            target_outs = model(target_batch)
        target_loss = target_outs.loss
        meta_loss_Str_K += target_loss

        combined_batch = AttrDict(xc=batch.x, yc=batch.y, xt=batch.x, yt=batch.y, x=batch.x, y=batch.y)
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            combined_outs = model(combined_batch, num_samples=args.train_num_samples)
        else:
            combined_outs = model(combined_batch)
        combined_loss = combined_outs.loss
        meta_loss_S_K += combined_loss

        optimizer.zero_grad()
        meta_loss_Str_K.backward(retain_graph=True)
        u_g1 = [deepcopy(p.grad) for p in model.parameters()]

        optimizer.zero_grad()
        meta_loss_S_K.backward()
        u_g2 = [deepcopy(p.grad) for p in model.parameters()]

        g_incohen.append(cal_grad_incohence(u_g1, u_g2))
        g_inco_mean = np.mean(g_incohen)
        outer_g_inco_bound += g_inco_mean * args.lr * temperature / 2
        inco_bound = np.sqrt((outer_g_inco_bound) / (args.train_batch_size * args.num_steps * args.eval_num_batches))

        if args.sgld == 1:
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.normal(0, (optimizer.param_groups[0]['lr'] / args.temperature) ** 0.5, size=param.size()).cuda()
                    param.add_(noise)

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            line += f"   inco_bound: {inco_bound} "

            # Add DSR related logs if enabled
            if args.use_dsr and len(dsr_values) > 0:
                dsr_avg = sum(dsr_values[-10:]) / min(10, len(dsr_values))
                line += f"   dsr_avg: {dsr_avg:.4f} "
                
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(args, model)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)

def get_eval_path(args):
    path = osp.join(evalsets_path, 'gp')
    filename = f'{args.eval_kernel}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    return path, filename

def gen_evalset(args):
    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_points=args.max_num_points,
            device='cuda'))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def eval(args, model):

    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


def eval_all_metrics(args, model):

    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
    model.load_state_dict(ckpt.model)
    if args.eval_logfile is None:
        eval_logfile = f'eval_{args.eval_kernel}'
        if args.t_noise is not None:
            eval_logfile += f'_tn_{args.t_noise}'
        eval_logfile += f'_all_metrics'
        eval_logfile += '.log'
    else:
        eval_logfile = args.eval_logfile
    filename = os.path.join(args.root, eval_logfile)
    logger = get_logger(filename, mode='w')

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval_all_metrics":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        ravgs = [RunningAverage() for _ in range(4)]
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
                ll = model(batch, num_samples=args.eval_num_samples)
            elif args.model in ["tnpa", "tnpnd"]:
                outs = model.predict(
                    batch.xc, batch.yc, batch.xt,
                    num_samples=args.eval_num_samples
                )
                ll = model(batch)
            else:
                outs = model.predict(batch.xc, batch.yc, batch.xt)
                ll = model(batch)

            mean, std = outs.loc, outs.scale

            if mean.dim() == 4:
                var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                std = var.sqrt().squeeze(0)
                mean = mean.mean(dim=0).squeeze(0)
            
            mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
            yt = batch.yt.squeeze().cpu().numpy().flatten()

            acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
            calibration = uct.metrics.get_all_average_calibration(mean, std, yt, num_bins=100, verbose=False)
            sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
            scoring_rule = {'tar_ll': ll.tar_ll.item()}

            batch_metrics = [acc, calibration, sharpness, scoring_rule]
            for i in range(len(batch_metrics)):
                ravg, batch_metric = ravgs[i], batch_metrics[i]
                for k in batch_metric.keys():
                    ravg.update(k, batch_metric[k])

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    
    line += '\n'

    for ravg in ravgs:
        line += ravg.info()
        line += '\n'

    if logger is not None:
        logger.info(line)

    return line


def plot(args, model):
    seed = args.plot_seed
    num_smp = args.plot_num_samples

    if args.mode == "plot":
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
        model.load_state_dict(ckpt.model)
    model = model.cuda()

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    kernel = RBFKernel()
    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)

    xp = torch.linspace(-2, 2, 200).cuda()
    batch = sampler.sample(
        batch_size=args.plot_batch_size,
        num_ctx=args.plot_num_ctx,
        num_tar=args.plot_num_tar,
        device='cuda',
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Nc = batch.xc.size(1)
    Nt = batch.xt.size(1)

    model.eval()
    with torch.no_grad():
        if args.model in ["np", "anp", "bnp", "banp"]:
            outs = model(batch, num_smp, reduce_ll=False)
        else:
            outs = model(batch, reduce_ll=False)
        tar_loss = outs.tar_ll  # [Ns,B,Nt] ([B,Nt] for CNP)
        if args.model in ["cnp", "canp", "tnpd", "tnpa", "tnpnd"]:
            tar_loss = tar_loss.unsqueeze(0)  # [1,B,Nt]

        xt = xp[None, :, None].repeat(args.plot_batch_size, 1, 1)
        if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd"]:
            pred = model.predict(batch.xc, batch.yc, xt, num_samples=num_smp)
        else:
            pred = model.predict(batch.xc, batch.yc, xt)
        
        mu, sigma = pred.mean, pred.scale

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size//4, 1)
        ncols = min(4, args.plot_batch_size)
        _, axes = plt.subplots(nrows, ncols,
                figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                        alpha=max(0.5/args.plot_num_samples, 0.1))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue',
                        alpha=max(0.2/args.plot_num_samples, 0.02),
                        linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}', zorder=mu.shape[0] + 1)
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}',
                       zorder=mu.shape[0] + 1)
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}')
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}')
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")

    plt.tight_layout()

    save_dir_1 = osp.join(args.root, f"plot_num{num_smp}-c{Nc}-t{Nt}-seed{seed}-{args.start_time}.jpg")
    file_name = "-".join([args.model, args.expid, f"plot_num{num_smp}",
                          f"c{Nc}", f"t{Nt}", f"seed{seed}", f"{args.start_time}.jpg"])
    if args.expid is not None:
        save_dir_2 = osp.join(results_path, "gp", "plot", args.expid, file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot", args.expid)):
            os.makedirs(osp.join(results_path, "gp", "plot", args.expid))
    else:
        save_dir_2 = osp.join(results_path, "gp", "plot", file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot")):
            os.makedirs(osp.join(results_path, "gp", "plot"))
    plt.savefig(save_dir_1)
    plt.savefig(save_dir_2)
    print(f"Evaluation Plot saved at {save_dir_1}\n")
    print(f"Evaluation Plot saved at {save_dir_2}\n")

if __name__ == '__main__':
    main()
