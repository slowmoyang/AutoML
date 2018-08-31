from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from controller import Controller
from childmodel import SearchSpace
from losses import REINFORCELossWithEMA

from dataset import JetDataset


class Trainer(object):
    def __init__(self,
                 num_nodes=12,
                 controller_lr=0.00035,
                 search_space_lr=20.0):

        ###############################################
        # NOTE Dataset
        ###############################################
        dset = JetDataset(
            "/data/slowmoyang/QGJets/root_100_200/2-Refined/dijet_test_set.root",
            max_len=32)
        self._train_loader = DataLoader(dset, batch_size=128)
        self._valid_loader = DataLoader(dset, batch_size=128)
        self._test_loader = DataLoader(dset, batch_size=128)

        # FIXME
        self._train_iter = itertools.cycle(self._train_loader)
        self._valid_iter = itertools.cycle(self._valid_loader)

        ######################################################
        # NOTE SearchSpace
        ##################################################
        self._operations = [
            "relu", "tanh", "sigmoid", "identity"
        ]
        self._num_operations = len(self._operations)

        self._search_space = SearchSpace(
            input_size=100,
            num_nodes=12,
            hidden_size=100,
            operations=self._operations
        )

        self._controller = Controller(
            num_nodes=num_nodes,
            num_operations=self._num_operations
        )

        #############################
        #
        ##############################
        self._search_space_loss = nn.BCEWithLogitsLoss()
        self._controller_loss = REINFORCELossWithEMA()

        self._controller_optim = optim.Adam(
            params=self._controller.parameters(),
            lr=controller_lr)

        self._search_space_optim = optim.SGD(
            params=self._search_space.parameters(),
            lr=search_space_lr)


        ####################################
        #
        ###############################################
        self._controller_lr = controller_lr
        self._search_space_lr = search_space_lr


    def train(self):
        # TODO

        self._search_space.reset_parameters(mode="search")
        # self._search_space.reset_parameters(mode="derivation")

        many_iterations = 10
        for _ in range(many_iterations):
            self._train_search_space()
            self._train_controller()


    def _train_search_space(self):
        self._search_space.train()
        self._controller.eval()

        dag = self._controller(with_log_prob=False)
        self._search_space.dag = dag

        # FIXME while not convegence
        # we train the shared parameters of the child models during a entrie
        # pass through the training data.
        for batch in self._train_loader:
            input, target = batch["x"], batch["y"]
            # input, target = input.cuda(), target.cuda()

            self._search_space_optim.zero_grad()

            output = self._search_space(input)
            loss = self._search_space_loss(output, target)

            loss.backward()

            # TODO self._search_space_max_grad_norm = 0.25
            nn.utils.clip_grad_norm(
                self._search_space,
                self._max_grad_norm)

            self._search_space_optim.step()

    def _train_controller(self):
        """Training the controller parameters
        Fix omega and update the policy parameters theta
        the gradient is computed using REINFORCE, with a moving average baseline
        to reduce variance.
        """
        self._search_space.eval()
        self._controller.train()

        for _ in range(self._controller_max_steps):
            dag, log_prob = self._controller()
            self._search_space.dag = dag

            batch = self._valid_iter.next()            
            input, target = batch["x"], batch["y"]
            # input, target = input.cuda(), target.cuda()

            self._controller_optim.zero_grad()

            output = self._search_space(input)

            reward = self._accuracy(output, target)
            loss = self._controller_loss(
                reward=reward,
                log_prob=log_prob)

            loss.backward()

            self._controller_optim.step()

    def derieve(self):
        """Deriving Architectures
        Step 1: Sample several models from the trained policy.
        Step 2: Compute its reward on a single minibatch sampled from the
                validation set.
        Step 3: Take only the model with the highest reward.
        Step 4: Re-train from scratch.
        """

        dags = self._controller.sample(num_samples=self._num_samples)

        best_dag = None
        best_reward = 0.0

        for dag in dags:
            self._search_space.dag = dag

            batch = self._test_batch.iter()
            input, target = batch["x"], batch["y"]

            output = self._search_space(input)

            reward = self._accuracy(output, target) 

            if reward > best_reward:
                best_reward = reward
                best_dag = dag

        # 
        self._search_space.train()
        self._controller.eval()

        self._search_space = dag
        self._search_space.reset_parameters(mode="retrain")

        for batch in self._train_loader:
            input, target = batch["x"], batch["y"]
            # input, target = input.cuda(), target.cuda()

            self._search_space_optim.zero_grad()

            output = self._search_space(input)
            loss = self._search_space_loss(output, target)
            loss.backward()

            # TODO self._search_space_max_grad_norm = 0.25
            nn.utils.clip_grad_norm(
                self._search_space,
                self._max_grad_norm)

            self._search_space_optim.step()

        # TODO test a best child model and save it

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.derieve()
    trainer.retrain()

