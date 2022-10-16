import torch
import torch.nn as nn
import numpy as np
import math


class Mish(nn.Module):
    """
    Mish activation function is proposed in "Mish: A Self
    Regularized Non-Monotonic Neural Activation Function"
    paper, https://arxiv.org/abs/1908.08681.
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


# ========================= #
#   GoogLeNet from scratch  #
# ========================= #

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            return self.relu(self.batch_norm(self.conv(x)))

        else:
            return self.relu(self.conv(x))


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, ch3x3reduce, out_ch3x3, ch5x5reduce, out_ch5x5, pool_proj, use_bn=True):
        super(Inception_block, self).__init__()

        self.branch1 = BasicConv2d(in_channels, out_1x1,
                                   use_bn=use_bn, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3reduce,
                        use_bn=use_bn, kernel_size=1),

            BasicConv2d(ch3x3reduce, out_ch3x3, kernel_size=3,
                        use_bn=use_bn, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5reduce,
                        use_bn=use_bn, kernel_size=1),

            BasicConv2d(ch5x5reduce, out_ch5x5, use_bn=use_bn,
                        kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj,
                        use_bn=use_bn, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, use_bn=True):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)

        self.conv = BasicConv2d(in_channels, 128,
                                use_bn=use_bn, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class GoogLeNet(nn.Module):
    """
    Main GoogLeNet class body
    """

    def __init__(self, aux_logits=True, use_bn=True,
                 num_classes=10, init_weights=True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3)
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = BasicConv2d(in_channels=64,
                                 out_channels=192,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32, use_bn=use_bn)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64, use_bn=use_bn)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64, use_bn=use_bn)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64, use_bn=use_bn)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64, use_bn=use_bn)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64, use_bn=use_bn)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128, use_bn=use_bn)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128, use_bn=use_bn)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128, use_bn=use_bn)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes, use_bn=use_bn)
            self.aux2 = InceptionAux(528, num_classes, use_bn=use_bn)
        else:
            self.aux1 = self.aux2 = None

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


# =========================== #
#   MobileNetV2 from scratch  #
# =========================== #

class ConvBN(nn.Sequential):
    def __init__(self, inp, output, stride):
        super(ConvBN, self).__init__(
            nn.Conv2d(inp, output, kernel_size=3,
                      stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True),
        )


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class ConvBN_1x1(nn.Sequential):
    def __init__(self, inp, output):
        super(ConvBN_1x1, self).__init__(
            nn.Conv2d(inp, output, kernel_size=1,
                      stride=1, padding=0, bias=False),

            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=10, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [ConvBN(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(ConvBN_1x1(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def mobilenet_v2(pretrained=True):
        model = MobileNetV2(width_mult=1)

        if pretrained:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            state_dict = load_state_dict_from_url(
                'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
            model.load_state_dict(state_dict)
        return model


def googlenet_test():
    # N = 3 (Mini batch size)
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet(aux_logits=True, use_bn=False,
                      num_classes=10, init_weights=True)
    print(model(x)[2].shape)


def mobilenev2_test():
    net = MobileNetV2()
    output = net(torch.randn(4, 3, 224, 224))
    assert output.shape == (4, 10)
    print('Success')


if __name__ == "__main__":
    googlenet_test()
    mobilenev2_test()
