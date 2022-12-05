from utils import get_bins, generate_exp_name, process_args, plot_gradtraj, plotprediction, plot_layer12_feats, grad_angles, plotlosses
from model import MLP_Mean_Field_Init, batched_forward, eval_L2, eval_L2_CE, CEupdate, L2update, L2, CE
from toydata import BalancedDataset, multiscale_triangle, onehot
import numpy as np
from metrics import RegLogger, ClassLogger
from tqdm import tqdm
import tensorflow.config as tfconfig
from jax import jit, grad
import jax.random as jrandom
import jax.numpy as jnp
import pickle, os, time, json, argparse
import matplotlib.pyplot as plt

tfconfig.set_visible_devices([], device_type='GPU')


def train_regression(ds, params, args, tol=1e-7):

    el2, elogits = eval_L2(params, ds.xe, ds.ye)
    tl2, tlogits = eval_L2(params, ds.x, ds.y)
    logger = RegLogger(params, el2.item(), tl2.item(), max_count=10000, tol=tol)

    args['evalL2']= [float(el2.item())]
    args['trainL2'] = [float(tl2.item())]

    if args['max_epochs'] < 100:
        print_marker = args['max_epochs']
    else:
        print_marker = args['max_epochs'] / 100


    get_grads = jit(grad(L2))
    # store the first 500 and last 500 gradients
    grad_traj = []
    tail = []


    for i in tqdm(range(args['max_epochs'])):
        for (x,y) in ds.batch_and_shuffle():

            params = L2update(params, x, y, args['lr'])

        # evaluate model at the end of each epoch 
        el2, elogits = eval_L2(params, ds.xe, ds.ye)
        tl2, tlogits = eval_L2(params, ds.x, ds.y)

        args['trainL2'].append(float(el2.item()))
        args['evalL2'].append(float(tl2.item()))

        if (i+1) % print_marker == 0:
            tqdm.write(f"L2 Train: {tl2:.8f}, L2 Eval {el2:.8f}")

        # store the gradients for the first 1000 epochs to asses the goodfellow line hypothesis
        grads = get_grads(params, ds.x, ds.y)
        if i < 1000:
            grad_traj.append(np.concatenate([g.flatten() for g in grads]).tolist())
        stop = logger.update(params, el2.item(), tl2.item(), i + 1)
        if stop:
            break

    args['tepochs'] = int(i + 1)
    logger.print_best(args)
    args = logger.update_args(args)
    # grad_traj += tail

    return logger, args, np.array(grad_traj)


def train_classification(ds, params, args, tol=1e-7):
    assert args['C'] == args['dC']

    # evaluate on test set
    el2, ece, eacc, elogits = eval_L2_CE(params, ds.xe, ds.ye, ds.yce, ds.yche, args['midpoints'])
    args['evalL2'] = [float(el2.item())]
    args['evalCE'] = [float(ece.item())]
    args['evalACC'] = [float(eacc.item())]

    # evaluate on train set
    tl2, tce, tacc, tlogits = eval_L2_CE(params, ds.x, ds.y, ds.yc, ds.ych, args['midpoints'])
    args['trainL2'] = [float(tl2.item())]
    args['trainCE'] = [float(tce.item())]
    args['trainACC']= [float(tacc.item())]

    print(f"CE Train: {tce:.6f}, CE Eval: {ece:.6f}, L2 Train: {tl2:.46},\
          L2 Eval {el2:.6f}, Acc Train: {tacc:.6f}, Acc Eval {eacc:.6f}")


    logger = ClassLogger(params, el2.item(), tl2.item(), eacc.item(), tacc.item(), tce.item(),
                         max_count=10000, tol=tol)
    print("Training:")


    if args['max_epochs'] < 100:
        print_marker = args['max_epochs']
    else:
        print_marker = args['max_epochs'] / 100


    get_grads = jit(grad(CE))
    # store the first and last 500 gradients for calculating grad trajectory
    grad_traj = []
    tail = []


    for i in tqdm(range(args['max_epochs'])):
        # for j in range(args['nbs']):
        for (x, y, yc, ych) in ds.batch_and_shuffle():

            params= CEupdate(params, x, ych, args['lr'])


        el2, ece, eacc, elogits = eval_L2_CE(params, ds.xe, ds.ye, ds.yce, ds.yche, args['midpoints'])
        args['evalL2'].append(float(el2.item()))
        args['evalCE'].append(float(ece.item()))
        args['evalACC'].append(float(eacc.item()))

        tl2, tce, tacc, tlogits = eval_L2_CE(params, ds.x, ds.y, ds.yc, ds.ych, args['midpoints'])
        args['trainL2'].append(float(tl2.item()))
        args['trainCE'].append(float(tce.item()))
        args['trainACC'].append(float(tacc.item()))


       # evaluate model every time marker is reached
        if (i+1) % print_marker == 0:

            tqdm.write(f"CE Train: {tce:.4f}, CE Eval: {ece:.4f}, L2 Train: {tl2:.4f}, \
                       L2 Eval {el2:.4f}, Acc Train: {tacc:.4f}, Acc Eval {eacc:.4f}")

        grads = get_grads(params, ds.x, ds.ych)
        if i < 1000:
            grad_traj.append(np.concatenate([g.flatten() for g in grads]).tolist())
        stop = logger.update(params, el2.item(), tl2.item(),
                             eacc.item(), tacc.item(), tce.item(), i + 1)
        if stop:
            break



    args['tepochs'] = int(i + 1)
    logger.print_best(args)
    args = logger.update_args(args)

    return logger, args, np.array(grad_traj)



def main(args):

    # if the experiment has already be run the reload
    if args['reload_path'] != "":
        assert os.path.isdir("experiments/" + args['exp_name'])
        # load best parameters
        with open(os.path.join(args['reload_path'], 'weights.pickle'), 'rb') as f:
            best_params = pickle.load(f)
        best_params = [jnp.array(w) for w in best_params]

        with open(os.path.join(args['reload_path'],'grad_traj.npy'), 'rb') as f:
            grad_traj = np.load(f)
        # load arguments from the experiment
        with open(os.path.join(args['reload_path'], 'args.json'), 'r') as f:
            args = json.load(f)

        dstate, tstate = np.random.RandomState(args['data_seed']), np.random.RandomState(args['train_seed'])
        reg = True if args['C'] == 1 else False
        # reload the data set 
        ds = BalancedDataset(multiscale_triangle, args['nbs'], args['bs'],
                             args['ne'], args['cbins'], dstate, reg=reg)
        args['save'] = False
        try:
            print("BEST RMSE OBTAINED:", np.sqrt(args['best_el2']))
        except:
            pass

    # else run experiment
    else:

        start_time = time.time()

        key = jrandom.PRNGKey(args['train_seed'])
        dstate, tstate = np.random.RandomState(args['data_seed']), np.random.RandomState(args['train_seed'])

        # generate model parameters
        print("initializing model")
        params = MLP_Mean_Field_Init(args['layer_sizes'], key)

        # regression  problem 
        if args['C'] == 1:
            # generate regression data
            ds = BalancedDataset(multiscale_triangle, args['nbs'], args['bs'],
                                 args['ne'], args['cbins'], dstate, reg=True)

            # train the model
            logger, args, grad_traj = train_regression(ds, params, args)

        else:
            # generate classification data
            ds = BalancedDataset(multiscale_triangle, args['nbs'], args['bs'],
                                 args['ne'], args['cbins'], dstate, reg=False)

            logger, args, grad_traj = train_classification(ds, params, args)

        # elapsed time in minutes 
        elapsed_time = (time.time() - start_time) / 60
        args['elapsed_time'] = float(elapsed_time)
        print(f"Elapsed train time (mins): {elapsed_time}")

        best_params = logger.log['el2'][2]

    # plot the angles between the gradients
    grad_angles(args, grad_traj)
    # plot learnt function
    plotprediction(ds, best_params, args)
    # plot position of features
    plot_layer12_feats(ds, args, best_params)
    # plot eigenvalues of gradient cov matrix
    plot_gradtraj(args, grad_traj)

   # plot losses
    plotlosses(args['trainL2'], args, title="L2 Train", yscale='log')
    plotlosses(args['evalL2'], args, title="L2 Eval", yscale='log')
    if args['C'] > 1:
        plotlosses(args['trainCE'], args, title="CE Train", yscale='log')
        plotlosses(args['evalCE'], args, title="CE Eval", yscale='log')


    # save parameters / weights for the experiment
    if args['save']:
        print("Saving Params & Logs...")
        args['midpoints'] = args['midpoints'].tolist()
        json_dict = json.dumps(args, indent=4)
        ppath = os.path.join(args['save_dir'], "args.json")
        with open(ppath, "w") as f:
            f.write(json_dict)
        # wpath = os.path.join(args['save_dir'], 'weights.npy')
        wpath = os.path.join(args['save_dir'], 'weights.pickle')
        bparams = [p.tolist() for p in best_params]
        with open(wpath, 'wb') as f:
            pickle.dump(bparams, f)

        with open(os.path.join(args['save_dir'], "grad_traj.npy"), 'wb') as f:
            np.save(f, grad_traj)

        print("Params & Logs saved!")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--layer_sizes", default="2,100,1", type=str,
                        help="layer sizes for the MLP, the final is the number of bins")

    parser.add_argument("--lr", type=float,
                        default=0.1,
                        help="learning rate (default 0.1).")

    parser.add_argument("--bs", type=int,
                        default=250,
                        help="batch size (default 128)")

    parser.add_argument("--nbs", type=int,
                        default=1,
                        help="no of batches for finite train dataset")

    parser.add_argument("--dC", type=int,
                        default=50,
                        help="number of classes to balance data into (default 25)")

    parser.add_argument('--ne', type=int,
                        default=250, help="number of evaluation data points")

    parser.add_argument("--max_epochs", type=int,
                        default=50000,
                        help="Maximum number of epochs to train over dataset. \
                        If tolerance is reached then will end")

    parser.add_argument("--train_seed", type=int,
                        default=2384,
                        help="random seed for experiment training / model weights ect")

    parser.add_argument("--data_seed", type=int, default=1214,
                        help='random seed for generating the data sets (train and eval)')

    parser.add_argument("--save", help='save the plots and model',
                        default=False,
                        action='store_true')

    parser.add_argument("--exp_name", default='', type=str,
                        help="name of experiment folder for saving")

    parser.add_argument('--reload_path', default='', help='path to experiment for reload')


    args = parser.parse_args()
    args = vars(args)
    args = process_args(args)
    main(args)



