from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
from torch import nn
import torch.nn.functional as F


class StackedRNNCellBase(nn.Module):
    def __init__(self,
                 cell_type,
                 input_size,
                 hidden_size,
                 bias=True,
                 num_cells=2):
        super(StackedRNNCellBase, self).__init__()

        stacked_cells = []
        for cell_id in range(num_cells):
            input_size = input_size if cell_id == 0 else hidden_size

            cell = cell_type(
                input_size=input_size,
                hidden_size=hidden_size,
                bias=bias)

            stacked_cells.append(cell)
        
        self.stacked_cells = nn.ModuleList(stacked_cells)

        self._input_size = input_size
        self._hidden_size = hidden_size


class StackedLSTMCell(StackedRNNCellBase):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 num_cells=2):
        super(StackedLSTMCell, self).__init__(
            cell_type=nn.LSTMCell,
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_cells=num_cells)
        
    def forward(self,
                input,
                (stacked_h_0, stacked_c_0)):

        stacked_h_1 = []
        stacked_c_1 = []
        for cell, h_0, c_0 in zip(self.stacked_cells, stacked_h_0, stacked_c_0):
            h_1, c_1 = cell(input, (h_0, c_0))
            stacked_h_1.append(h_1)
            stacked_c_1.append(c_1)
            input = h_1

        stacked_h_1 = torch.stack(stacked_h_1)
        stacked_c_1 = torch.stack(stacked_c_1)

        return (stacked_h_1, stacked_c_1)


class StackedGRUCell(StackedRNNCellBase):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 num_cells=2):
        super(StackedGRUCell, self).__init__(
            cell_type=nn.GRUCell,
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_cells=num_cells)
        
    def forward(self,
                input,
                stacked_h_0):

        stacked_h_1 = []
        for cell, h_0 in zip(self.stacked_cells, stacked_h_0):
            h_1 = cell(input, h_0)
            stacked_h_1.append(h_1)
            input = h_1

        stacked_h_1 = torch.stack(stacked_h_1)
        return stacked_h_1


class HighwayConnection(nn.Module):
    def __init__(self, in_features, out_features, activation="relu"):
        super(HighwayConnection, self).__init__()

        # FIXME
        assert in_features == out_features
        
        self._linear = nn.Linear(
            in_features=in_features,
            out_features=out_features)
        
        self._transform_gate = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out_features),
            nn.Sigmoid())
        
        self.activation = activation
    
    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, activation_):
        if isinstance(activation_, str):
            if hasattr(torch, activation_):
                activation_ = getattr(torch, activation_)
            elif hasattr(F, activation_):
                activation_ = getattr(F, activation_)
            else:
                raise ValueError
        elif activation_ is None:
            pass
        else:
            raise ValueError            

        self._activation = activation_
    
    def forward(self, x):
        h = self._linear(x)
        if self._activation is not None:
            h = self._activation(h)
        t = self._transform_gate(x)
        # Carry gate
        c = 1 - t
        
        y = h * t + x * c
        return y


class MultiInputHighwayConnection(HighwayConnection):
    def __init__(self,
                 in_features_list,
                 activation="relu"):

        super(MultiInputHighwayConnection, self).__init__()

        # FIXME
        assert in_features == out_features

        linears = []
        transform_gate_linear = []
        for in_features in in_features_list:        
            linears.append(nn.Linear(in_features, in_features))
            
        self._linears = nn.ModuleList(linears)


        self._transform_gate = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out_features),
            nn.Sigmoid())
        
        self.activation = activation

    def forward(self, x, h_prev):
        h_curr = self._linear(x, h_prev)

        if self._activation is not None:
            h_curr = self._activation(h_curr)

        # Transform gate
        t = self._transform_gate(x, h_prev)
        # Carry gate
        c = 1 - t
        y = h_curr * t + x * c
        return y

    



class VariationalDropout(nn.Module):
    def __init__(self):
        raise NotImplementedError



def _test_stacked_rnn_cell():
    num_cells = 2
    input_size = 100
    hidden_size = 100

    input = torch.randn(1, input_size)
    stacked_hidden_states = torch.randn(num_cells, 1, hidden_size)
    stacked_cell_states = torch.randn(num_cells, 1, hidden_size)

    print("Test StackedLSMTCell")
    stacked_lstm_cell = StackedLSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_cells=num_cells)

    print(stacked_lstm_cell)

    stacked_hidden_states, stacked_cell_states =  stacked_lstm_cell(
        input,
        (stacked_hidden_states,
        stacked_cell_states))

    print(stacked_hidden_states.shape)


def _test_highway_connection():
    hc = HighwayConnection(10, 10)
    x = torch.randn(512, 10)
    y = hc(x)
    hc.activation = "relu"
    y = hc(x)
    print(y.shape)


if __name__ == "__main__":
    _test_stacked_rnn_cell()
    _test_highway_connection()
