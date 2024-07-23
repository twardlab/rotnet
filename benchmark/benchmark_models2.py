import os
import sys
import importlib

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import moment_kernels as mk
importlib.reload(mk)

import torch # type: ignore
import torch.nn as tnn # type: ignore
import torch.nn.functional as F # type: ignore

import escnn.nn as enn # type: ignore
import escnn.gspaces as gspaces # type: ignore

class TorchConvBlock(tnn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size, padding):
        super(TorchConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # If the input and output channels are the same, then the stride is 1, do not downsample
        if in_channels == out_channels:
            stride = 1
        else:
            stride = 2

        self.block = tnn.Sequential(
            tnn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride),
            tnn.BatchNorm2d(out_channels),
            tnn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class VanillaCNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 5):
        super(VanillaCNN, self).__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** i) for i in range(num_layers)]
        self.layers_list = tnn.ModuleList()

        # Create the layers (according to the number of layers specified)
        for i in range(num_layers):
            self.layers_list.append(TorchConvBlock(self.stages[i], self.stages[i + 1], kernel_size, padding))

        self.layers = tnn.Sequential(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class ECNNConvBlock(enn.EquivariantModule):
    def __init__(self, in_type : enn.FieldType, out_type : enn.FieldType, group, kernel_size, padding):
        super(ECNNConvBlock, self).__init__()
        self.in_type = in_type
        self.out_type = out_type

        # If the input and output channels are the same, then the stride is 1, do not downsample
        if in_type.size == out_type.size:
            stride = 1
        else:
            stride = 2

        # extract out_type names and group them
        out_fields_names = [name for name in out_type.fields_names()]
        grouped_fields = out_type.group_by_labels(out_fields_names)

        # initialize an empty list for vector and non-vector fields
        vector_fields = []
        non_vector_fields = []

        # iterate over the grouped fields and append them to the appropriate list
        for field_name, field in grouped_fields.items():

            # irrep_0 is simply the scalar representation
            if field_name in ['irrep_0', 'regular']:
                non_vector_fields += field
            else:
                vector_fields += field

        # create the appropriate field types
        vector_type = enn.FieldType(group, vector_fields) if len(vector_fields) > 0 else None
        non_vector_type = enn.FieldType(group, non_vector_fields) if len(non_vector_fields) > 0 else None

        # create the appropriate equivariant layers. Motivation: we must be very picky which nonlinearity to use with which field type. Vector fields require NormReLU, non-vector fields can use ReLU
        # Under mixed representations, we must use MultipleModule to handle the different types of nonlinearities accordingly. A consequence of this implementation is that in the model itself
        # we must first declare non-vector fields and then vector fields. This is because the MultipleModule will maintain the nonlinearities in the order they are declared.
        if vector_type is not None and non_vector_type is not None:
            norm_relu = enn.NormNonLinearity(vector_type)
            relu = enn.ReLU(non_vector_type)
            out_type = non_vector_type + vector_type
            self.nonlinearity = enn.MultipleModule(out_type, ['relu'] * len(non_vector_type) + ['norm_relu'] * len(vector_type), [(relu, 'relu'), (norm_relu, 'norm_relu')])
        elif vector_type is not None:
            norm_relu = enn.NormNonLinearity(vector_type)
            out_type = vector_type
            self.nonlinearity = enn.MultipleModule(out_type, ['norm_relu'] * len(vector_type), [(norm_relu, 'norm_relu')])
        elif non_vector_type is not None:
            relu = enn.ReLU(non_vector_type)
            out_type = non_vector_type
            self.nonlinearity = enn.MultipleModule(out_type, ['relu'] * len(non_vector_type), [(relu, 'relu')])

        self.block = enn.SequentialModule(enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, padding = padding, stride = stride),
                                          enn.GNormBatchNorm(out_type),
                                          self.nonlinearity
                                          )
        
    def forward(self, x):
        return self.block(x)
    
    def evaluate_output_shape(self, input_shape: tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

class TrivialECNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 4, N_equivariance : int = 5):
        super(TrivialECNN, self).__init__()
        self.in_type = None
        self.out_type = None
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** i) for i in range(num_layers)]
        self.layers_list = []

        # Equivariance to rotations of C_N
        self.r2_act = gspaces.rot2dOnR2(N = N_equivariance)

        for i in range(num_layers):

            # On the very first layer, we must specify that we want to use the trivial representation as the input * img_channels
            # On subsequent layers, the input type is simply the output of the previous layer
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
class TrivialIrrepECNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 4, N_equivariance : int = 4):
        super(TrivialIrrepECNN, self).__init__()
        self.in_type = None
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** i) for i in range(num_layers)]
        self.layers_list = []

        # Equivariance to rotations of C_N
        self.r2_act = gspaces.rot2dOnR2(N = N_equivariance)

        for i in range(num_layers):

            # The input of first layer should be trivial only. The output of last layer should be trivial only. Intermediate layers should be both trivial and irrep
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, (self.stages[i + 1] // 2) * [self.r2_act.trivial_repr] + (self.stages[i + 1] // 2) * [self.r2_act.irrep(1)])
            elif i == num_layers - 1:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, (self.stages[i + 1] // 2) * [self.r2_act.trivial_repr] + (self.stages[i + 1] // 2) * [self.r2_act.irrep(1)])
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
class RegularECNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 4, N_equivariance : int = 4):
        super(RegularECNN, self).__init__()
        self.in_type = None
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** i) for i in range(num_layers)]
        self.layers_list = []

        # Equivariance to rotations of C_N
        self.r2_act = gspaces.rot2dOnR2(N = N_equivariance)

        for i in range(num_layers):
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.regular_repr])
            elif i == num_layers - 1:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.regular_repr])
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x
    
def test_model(model: torch.nn.Module, device: torch.device, input_shape: tuple):
    model.to(device).eval()
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        y = model(x)
    return y

# count number of parameters
def count_parameters(model):
    model.train()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)