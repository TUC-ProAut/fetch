from torch import nn
import copy


class FCBlock(nn.Module):
    def __init__(self, opt, in_channels, out_channels, bias=False):
        super(FCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        lin = nn.Linear(in_channels, out_channels, bias=bias)
        
        layer = [lin]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + '1d')(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + '1d')(num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [lin, bn]

        if opt.activetype != 'None':
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(lin)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)


def FinalBlock(num_classes, opt, in_channels, bias=False):
    opt = copy.deepcopy(opt)
    if not opt.preact: opt.activetype = 'None'
    return FCBlock(opt=opt, in_channels=in_channels, out_channels=num_classes, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        
        layer = [conv]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype+'2d')(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype+'2d')(num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [conv, bn]

        if opt.activetype is not 'None':
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(conv)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)