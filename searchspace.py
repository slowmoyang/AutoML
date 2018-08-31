import torch
from torch import nn
import torch.nn.functional as F

from modules import HighwayConnection

# TODO Parameters Initialization
#  - uniformly in [-0.025, 0.025] during architecture search
#  - uniformly in [-0.04, 0.04] when we train a fixed architectures recommend
#    by the controller

# TODO VariationalDropout
# TODO tying word embeddings and softmax weights


class SearchSpace(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_nodes,
                 operations,
                 batch_first=False):
        super(SearchSpace, self).__init__() 

        connections = []
        for node_id in range(1, num_nodes):
            connections.append(nn.ModuleList([HighwayConnection(hidden_size, hidden_size) for prev_node_id in range(node_id)]))
        self._connections = nn.ModuleList(connections)

        # None means identity operation
        self._activated_connections = [None] * num_nodes

        self._operations = operations

        self._prev_node_ids = None
        self._operation_indices = None

        self._batch_first = batch_first
        self._batch_axis = 0 if batch_first else 1

    @property
    def architecture(self):
        return (self._prev_node_ids, self._operation_indices)

    @architecture.setter
    def architecture(self, architecture_):
        prev_node_ids, operation_indices = architecture_

        for node_id, (prev_node_id, op_idx) in enumerate(zip(prev_indices, operation_indices[1:]), 1):
            self._activated_connections[node_id] = self._connections[node_id][prev_node_id]
            self._activated_connections[node_id].activation = self._operations[op_idx]

    # recurrent cell
    def _cell(self, x, h_0):
        h_1 = self._activated_connections[0](x, h_0)

        loose_ends = []
        h_1 = layer(x, h_0)

        for node_id, layer in enumerate(self._activated_connections):
            h_1 = layer(h_1)

            if node_id not in self._prev_indices:
                loose_ends.append(h_1)

        loose_ends = torch.stack(loose_ends)
        output = loosed_ends.mean(dim=0)
        return output


    def forward(self, input, hidden_state=None, batch_first=False):
        if hidden_state is None:
            batch_size = input.size(self._batch_axis)
            hidden_state = torch.zeros(batch_size, self._hidden_size) 

        if batch_first:
            # (batch, seq_len, input_size) to (seq_len, batch, input_size)
            input = input.permute(1, 0, 2)
   
        output = []
        for x in input:
            hidden_state = self._cell(x, hidden_state)
            output.append(hidden_state)
        output = torch.stack(output)

        if batch_first:
            # (seq_len, batch, input_size) to (batch, seq_len, input_size)
            output = output.permute(1, 0, 2)

        return output 




def _test():
    search_space = SearchSpace(
        input_size=4,
        hidden_size=100,
        num_nodes=4,
        operations=["tanh", "identity", "sigmoid", "relu"])

if __name__ == "__main__":
    _test()
