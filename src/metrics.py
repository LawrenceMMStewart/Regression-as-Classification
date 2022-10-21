import numpy as np
import os

class RegLogger:
    def __init__(self, params, el2, tl2, max_count=1000, tol=1e-7):
        """
        logger for regression

        items are stored in dict e.g.

        key = evall2
        value = (best_epoch, best_value, best_params)
        """

        self.log = {}
        self.log['el2'] = (0, el2, [np.copy(p) for p in params])
        self.log['tl2'] = (0, tl2, [np.copy(p) for p in params])
        self.stop_counter = 0
        self.max_count = max_count
        self.tol = tol

    def update(self, params, el2, tl2, epoch):

        if tl2 + self.tol <= self.log['tl2'][1]:
            self.stop_counter = 0
            self.log['tl2'] = (epoch, tl2, [np.copy(p) for p in params])
        if el2 + self.tol <= self.log['el2'][1]:
            self.stop_counter = 0
            self.log['el2'] = (epoch, el2, [np.copy(p) for p in params])

        self.stop_counter += 1
        if self.stop_counter == self.max_count:
            print("No further improvement reached in training")
            return True

        return False

    def prepare_messages(self):
        self.messages = []
        self.messages.append(f'best train l2: {self.log["tl2"][1]} obtained on epoch {self.log["tl2"][0]}')
        self.messages.append(f'best eval l2: {self.log["el2"][1]} obtained on epoch {self.log["el2"][0]}')

    def update_args(self, args):
        args['best_tl2'] = self.log['tl2'][1]
        args['best_el2'] = self.log['el2'][1]
        return args

    def print_best(self, args):
        self.prepare_messages()
        for m in self.messages:
            print(m)
        if args['save']:
            path = os.path.join(args['save_dir'], "log.txt")
            with open(path, 'w+') as f:
                for m in self.messages:
                    f.write(m + '\n')




class ClassLogger:
    def __init__(self, params, el2, tl2, eacc, tacc, tce, max_count=1000, tol=1e-8):
        """
        logger for classification

        items are stored in dict e.g.

        key = evall2
        value = (best_epoch, best_value, best_params)
        """

        self.log = {}
        self.log['el2'] = (0, el2, [np.copy(p) for p in params])
        self.log['tl2'] = (0, tl2, [np.copy(p) for p in params])
        self.log['eacc'] = (0, eacc, [np.copy(p) for p in params])
        self.log['tacc'] = (0, tacc, [np.copy(p) for p in params])
        self.log['tce'] = (0, tce, [np.copy(p) for p in params])
        self.stop_counter = 0
        self.max_count = max_count
        self.tol = tol


    def update(self, params, el2, tl2, eacc, tacc, tce, epoch):

        if tl2 + self.tol <= self.log['tl2'][1]:
            self.stop_counter = 0
            self.log['tl2'] = (epoch, tl2, [np.copy(p) for p in params])
        if tacc > self.log['tacc'][1]:
            self.stop_counter = 0
            self.log['tacc'] = (epoch, tacc, [np.copy(p) for p in params])
        if el2 + self.tol <= self.log['el2'][1]:
            self.stop_counter = 0
            self.log['el2'] = (epoch, el2, [np.copy(p) for p in params])
        if eacc > self.log['eacc'][1]:
            self.stop_counter = 0
            self.log['eacc'] = (epoch, eacc, [np.copy(p) for p in params])
        if tce + self.tol <= self.log['tce'][1]:
            self.stop_counter = 0
            self.log['tce'] = (epoch, tce, [np.copy(p) for p in params])

        self.stop_counter += 1
        if self.stop_counter == self.max_count:
            print("No further improvement reached in training")
            return True

        return False

    def prepare_messages(self):
        self.messages = []
        self.messages.append(f'best train l2: {self.log["tl2"][1]} obtained on epoch {self.log["tl2"][0]}')
        self.messages.append(f'best eval l2: {self.log["el2"][1]} obtained on epoch {self.log["el2"][0]}')
        self.messages.append(f'best train acc: {self.log["tacc"][1]} obtained on epoch {self.log["tacc"][0]}')
        self.messages.append(f'best eval acc: {self.log["eacc"][1]} obtained on epoch {self.log["eacc"][0]}')

    def update_args(self, args):
        args['best_tl2'] = self.log['tl2'][1]
        args['best_el2'] = self.log['el2'][1]
        args['best_tacc'] = self.log['tacc'][1]
        args['best_eacc'] = self.log['eacc'][1]
        return args

    def print_best(self, args):
        self.prepare_messages()
        for m in self.messages:
            print(m)
        if args['save']:
            path = os.path.join(args['save_dir'], "log.txt")
            with open(path, 'w+') as f:
                for m in self.messages:
                    f.write(m + '\n')



