import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def lr_pad(x, padding=1):
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(1, 0), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(lr_pad(x, 1))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(lr_pad(out, 1))
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(lr_pad(out))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=(3, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        conv_list = []
        x = self.conv1(lr_pad(x, 3))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x); conv_list.append(x)
        x = self.layer2(x); conv_list.append(x)
        x = self.layer3(x); conv_list.append(x)
        x = self.layer4(x); conv_list.append(x)

        return conv_list


def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            LR_PAD(),
            nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//2, in_c//2, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//4),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//4, out_c, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, out_w):
        x = self.layer(x)

        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class HorizonNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, use_rnn):
        super(HorizonNet, self).__init__()
        self.backbone = backbone
        self.use_rnn = use_rnn
        if backbone == 'resnet18':
            self.feature_extractor = resnet18()
            _exp = 1
        elif backbone == 'resnet50':
            self.feature_extractor = resnet50()
            _exp = 4
        elif backbone == 'resnet101':
            self.feature_extractor = resnet101()
            _exp = 4
        else:
            raise NotImplementedError()

        _out_scale = 8
        self.stage1 = nn.ModuleList([
            GlobalHeightConv(64 * _exp, int(64 * _exp / _out_scale)),
            GlobalHeightConv(128 * _exp, int(128 * _exp / _out_scale)),
            GlobalHeightConv(256 * _exp, int(256 * _exp / _out_scale)),
            GlobalHeightConv(512 * _exp, int(512 * _exp / _out_scale)),
        ])
        self.step_cols = 4
        self.rnn_hidden_size = 512
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=_exp * 256,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(0.5)
            self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                    out_features=3 * self.step_cols)
            self.linear.bias.data[0::4].fill_(-1)
            self.linear.bias.data[4::8].fill_(-0.478)
            self.linear.bias.data[8::12].fill_(0.425)
        else:
            self.linear = nn.Sequential(
                nn.Linear(_exp * 256, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
            )
            self.linear[-1].bias.data[0::4].fill_(-1)
            self.linear[-1].bias.data[4::8].fill_(-0.478)
            self.linear[-1].bias.data[8::12].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False

    def freeze_bn(self):
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        x = self._prepare_x(x)
        iw = x.shape[3]
        block_w = int(iw / self.step_cols)
        conv_list = self.feature_extractor(x)
        down_list = []
        for x, f in zip(conv_list, self.stage1):
            tmp = f(x, block_w)  # [b, c, h, w]
            flat = tmp.view(tmp.shape[0], -1, tmp.shape[3])  # [b, c*h, w]
            down_list.append(flat)
        feature = torch.cat(down_list, dim=1)  # [b, c*h, w]

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        cor = output[:, :1]
        bon = output[:, 1:]

        return bon, cor
