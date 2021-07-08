import torch
import torch.nn as nn

class AbstractModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def get_pretrain_parameters(self):
        raise NotImplementedError()

    def get_train_parameters(self):
        raise NotImplementedError()

    def forward(self, data):
        raise NotImplementedError()


class DefaultCNNModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        # an affine operation: y = Wx + b
        self.c1 = 6
        self.c2 = 16
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.c1, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.c1, self.c2, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        num_categories = 64
        self.mlp_1 = nn.Sequential(
            nn.Linear(self.c2 * 4 * 4, 200),
            nn.LeakyReLU(),
            nn.Linear(200, num_categories),
            nn.LeakyReLU(),
        )
        self.mlp_head = nn.Linear(num_categories, 1)

    def get_pretrain_parameters(self):
        return self.parameters()

    def get_train_parameters(self):
        return self.mlp_head.parameters()

    def forward(self, data):
        data = data.view(data.shape[0], 1, 28, 28)
        data = self.cnn(data).view(-1, self.c2 * 4 * 4)
        data = self.mlp_1(data)
        data_min = data.min(1)[0].view(-1, 1)
        data_max = data.max(1)[0].view(-1, 1)
        data = (data - data_min) / (data_max - data_min) * 5.0
        return self.mlp_head(torch.softmax(data, 1))


class ModularMLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, activation=None,
                 last_layer=None, num_out=1):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0

        if len(layer_widths) == 0:
            layers = [nn.Linear(input_dim, num_out)]
        else:
            num_layers = len(layer_widths)
            layers = [nn.Linear(input_dim, layer_widths[0]), activation()]
            for i in range(1, num_layers):
                w_in = layer_widths[i-1]
                w_out = layer_widths[i]
                layers.extend([nn.Linear(w_in, w_out), activation()])
            layers.append(nn.Linear(layer_widths[-1], num_out))
        if last_layer:
            layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def initialize(self):
        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        final_layer = self.model[-1]
        nn.init.xavier_normal_(final_layer.weight.data, gain=1.0)
        nn.init.zeros_(final_layer.bias.data)

    def get_pretrain_parameters(self):
        return self.parameters()

    def get_train_parameters(self):
        return self.parameters()

    def forward(self, data):
        # print(data.shape)
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        return self.model(data)
