import numpy as np
import matplotlib.pyplot as plt
import glob, os, json


def sort_dict_by_value(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


if __name__ == "__main__":

    REG_PATH = "experiments/regexps"
    CLASS_PATH = "experiments/classexps"

    reg_results = {}
    class_results = {}

    # collect the regression results from a directory found at REG_PATH
    print("Loading regression results...")
    for dname in glob.glob(REG_PATH + "/reg_*"):
        with open(os.path.join(dname, "args.json"), 'rb') as f:
            args = json.load(f)
            reg_results[args['train_seed']] = args['best_el2']


    # collect the regression results from a directory found at REG_PATH
    print("Loading classification results...")
    for dname in glob.glob(CLASS_PATH + "/class_*"):
        with open(os.path.join(dname, "args.json"), 'rb') as f:
            args = json.load(f)
            class_results[args['train_seed']] = args['best_el2']

    reg_results = sort_dict_by_value(reg_results)
    class_results = sort_dict_by_value(class_results)


    reg_seeds = np.fromiter(reg_results.keys(), dtype='int32')
    reg_MSE = np.fromiter(reg_results.values(), dtype='float32')
    reg_RMSE = np.sqrt(reg_MSE)

    # worst regression model
    wrid = np.argmax(reg_MSE)


    class_seeds = np.fromiter(class_results.keys(), dtype='int32')
    class_MSE = np.fromiter(class_results.values(), dtype='float32')
    class_RMSE = np.sqrt(class_MSE)
    # worst classification model
    wcid = np.argmax(class_MSE)


    print("Regression results:")
    print(f"Worst: {np.max(reg_RMSE)}")
    print(f"Best: {np.min(reg_RMSE)}")
    print(f"Mean: {np.mean(reg_RMSE)}")
    print(f"Std dev {np.std(reg_RMSE)}")
    # 170925
    print(f"Worst model's seed {reg_seeds[wrid]}")
    print('\n')


    print("Classification results:")
    print(f"Worst: {np.max(class_RMSE)}")
    print(f"Best: {np.min(class_RMSE)}")
    print(f"Mean: {np.mean(class_RMSE)}")
    print(f"Std dev {np.std(class_RMSE)}")
    # 275772
    print(f"Worst model's seed {class_seeds[wcid]}")




    plt.figure()
    plt.style.use('seaborn-colorblind')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


    plt.ylabel('RMSE', fontsize=16)
    plt.boxplot([reg_RMSE, class_RMSE])
    plt.xticks([1, 2], ['Regression', 'Classification'], fontsize=22)
    plt.yticks(fontsize=18)
    # plt.yscale('log')
    plt.show()
    plt.close()

    import pdb; pdb.set_trace()



