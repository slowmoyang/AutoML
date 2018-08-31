from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from modules import StackedLSTMCell

# TODO initializes controller's parameters 

class Controller(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_operations,
                 hidden_size=100,
                 num_cells=2,
                 tanh_constant=2.5,
                 temperature=5.0):
        super(Controller, self).__init__()

        self._cell = StackedLSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_cells=num_cells)

        self._operation_decoder = nn.Linear(
            in_features=hidden_size,
            out_features=num_operations)

        # NOTE set-selecton type attention
        self._attention_prev = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)

        self._attention_curr = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)

        # decode from the hidden layer output to the index of previous node
        self._prev_node_id_decoder = nn.Linear(
            in_features=hidden_size,
            out_features=1)

        # 
        self._prev_node_id_embedding = nn.Embedding(
            num_embeddings=num_nodes-1,
            embedding_dim=hidden_size)

        self._operation_embedding = nn.Embedding(
            num_embeddings=num_operations,
            embedding_dim=hidden_size)

        # initial inputs
        # '1' means batch size
        self._init_inputs = torch.zeros(1, hidden_size) 
        self._init_hidden_state = torch.zeros(num_cells, 1, hidden_size)
        self._init_cell_state = torch.zeros(num_cells, 1, hidden_size)

        self._num_nodes = num_nodes
        self._num_operations = num_operations

        self._hidden_size = hidden_size
        self._num_cells = num_cells

        self._tanh_constant = tanh_constant
        self._temperature = temperature


    def forward(self, with_log_prob=True):
        '''auto-regression fashion
        ENAS design both the prev_node_id and the operations in RNN cells
        '''

        # prepare initial inputs
        inputs = self._init_inputs
        hidden_state = self._init_hidden_state
        cell_state = self._init_cell_state

        # NOTE for set-selection type attention
        prev = []

        prev_node_ids = []
        operations = []

        if with_log_prob:
            log_probs = []

        for node_id in range(self._num_nodes):
            ####################################################################
            # NOTE 1) samples a previous index
            ####################################################################
            # print("[{}-1] inputs: {}".format(node_id, inputs.shape))
            # print("[{}-1] hidden_state: {}".format(node_id, hidden_state.shape))
            # print("[{}-1] cell_state: {}".format(node_id, cell_state.shape))
            hidden_state, cell_state = self._cell(
                inputs, (hidden_state, cell_state))
            # hidden_state: (num_cells, batch, hidden_size)

            # TODO attention prev
            p = self._attention_prev(hidden_state[-1])
            prev.append(p)

            if node_id > 0:
                # NOTE set-selection type attention
                query = self._attention_curr(hidden_state[-1])
                query = query + torch.cat(prev[:-1], dim=0)
                query = torch.tanh(query)

                # sampling logits
                logits = self._prev_node_id_decoder(query)
                # logits: (# of previous nodes, 1)
                logits = logits.permute(1, 0)
                # logits: (1, # of previous nodes)

                logits = logits / self._temperature
                logits = self._tanh_constant * torch.tanh(logits)
                # NOTE
                # diff = torch.pow(node_id - torch.range(0, node_id), 2)
                # diff = diff.view(1, node_id)
                # diff = diff / 6.0
                # logits = logits - diff

                sampling_prob = F.softmax(logits, dim=-1)
                prev_node_id = sampling_prob.multinomial(num_samples=1)
                prev_node_ids.append(int(prev_node_id))

                # NOTE log probability
                if with_log_prob:
                    log_prob = F.log_softmax(logits, dim=-1)
                    log_prob = log_prob.gather(
                        dim=1,
                        index=prev_node_id)
                    log_probs.append(log_prob)

                inputs = self._prev_node_id_embedding(prev_node_id)
                inputs = inputs.view(1, -1)


            ####################################################################
            # NOTE 2) samples an activation function
            ####################################################################
            # print("[{}-2] inputs: {}".format(node_id, inputs.shape))
            # print("[{}-2] hidden_state: {}".format(node_id, hidden_state.shape))
            # print("[{}-2] cell_state: {}".format(node_id, cell_state.shape))

            hidden_state, cell_state = self._cell(
                inputs, (hidden_state, cell_state))

            logits = self._operation_decoder(hidden_state[-1])
            logits = logits / self._temperature
            logits = self._tanh_constant * torch.tanh(logits)

            sampling_prob = F.softmax(logits, dim=-1)
            op = sampling_prob.multinomial(num_samples=1)
            operations.append(int(op))

            if with_log_prob:
                log_prob = F.log_softmax(logits, dim=-1)
                log_prob = log_prob.gather(
                    dim=1,
                    index=op)
                log_probs.append(log_prob)

            inputs = self._operation_embedding(op)
            inputs = inputs.view(1, -1)

        if with_log_prob:
            log_prob = torch.cat(log_probs).squeeze().sum()
            return (prev_node_ids, operations), log_prob
        else:
            return (prev_node_ids, operations)


    def sample(self, num_samples):
        return [self(with_log_prob=False) for _ in range(num_samples)]


def _test():
    controller = Controller(num_nodes=12, num_operations=4)
    print(controller)
    print()
    architecture, log_prob = controller()
    prev_node_ids, operations = architecture
    print(prev_node_ids)
    print(operations)
    print("LogProb: {}".format(log_prob))
    architectures = controller.sample(num_samples=3)
    for each in architectures:
        print(each)

if __name__ == "__main__":
    _test()
