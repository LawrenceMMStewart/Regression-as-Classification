from train import main
from multiprocessing import Pool
import tensorflow.config as tfconfig
from utils import process_args
import numpy as np
tfconfig.set_visible_devices([], device_type='GPU')
np.random.seed(213497)



def get_args(train_seed, reg=True):
    args = {}
    args['bs'] = int(250)
    args['nbs'] = int(1)
    args['dC'] = int(50)
    args['ne'] = int(250)
    args['data_seed'] = int(52354)
    args['max_epochs'] = 2000000
    args['train_seed'] = train_seed
    args['reload_path'] = ''
    args['save'] =  True

    if reg:
        args['layer_sizes'] = '2,10000,1'
        args['lr']= float(0.04)
        args['exp_name'] = f'reg_{train_seed}'
    else:
        args['layer_sizes'] = '2,500,50'
        args['lr'] = float(0.07)
        args['exp_name'] = f'class_{train_seed}'

    args = process_args(args)
    return args




if __name__ == "__main__":

    NRUNS = 30
    Dseeds = np.random.randint(1,999999, size=(2 * NRUNS))
    Dseeds = np.unique(Dseeds)[:NRUNS]
    print(Dseeds)

    REG = False
    all_args = [get_args(int(ts),
                         reg=REG) for ts in Dseeds]

    NWORKERS = 4
    with Pool(NWORKERS) as pool:
        pool.map(main, all_args)

    REG = True
    all_args = [get_args(int(ts),
                         reg=REG) for ts in Dseeds]

    NWORKERS = 4
    with Pool(NWORKERS) as pool:
        pool.map(main, all_args)

