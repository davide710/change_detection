import torch
import torch.nn as nn


def yolo_params(version):
    if version == 'n':
        return 1/3, 1/4, 2
    elif version == 's':
        return 1/3, 1/2, 2
    elif version == 'm':
        return 2/3, 3/4, 1.5
    elif version == 'l':
        return 1, 1, 1
    elif version == 'x':
        return 1, 1.25, 1


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut
    
    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x += x_in
        return x
    
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks

        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([Bottleneck(self.mid_channels, self.mid_channels) for _ in range(num_bottlenecks)])
        self.conv2 = Conv((num_bottlenecks + 2)*self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x[:, :x.shape[1] // 2, :, :], x[:, x.shape[1] // 2:, :, :]
        outputs = [x1, x2]
        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)
            outputs.insert(0, x1)
        outputs = torch.cat(outputs, dim=1)
        out = self.conv2(outputs)
        return out
    
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2, dilation=1, ceil_mode=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x = torch.cat((x, x1, x2, x3), dim=1)
        x = self.conv2(x)
        return x
    
class Backbone(nn.Module):
    def __init__(self, version, in_channels=1, shortcut=True):
        super().__init__()
        d, w, r = yolo_params(version)
        self.conv_0 = Conv(in_channels, int(64 * w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1)

        self.c2f_2 = C2f(int(128 * w), int(128 * w), num_bottlenecks=int(3 * d), shortcut=True)
        self.c2f_4 = C2f(int(256 * w), int(256 * w), num_bottlenecks=int(6 * d), shortcut=True)
        self.c2f_6 = C2f(int(512 * w), int(512 * w), num_bottlenecks=int(6 * d), shortcut=True)
        self.c2f_8 = C2f(int(512 * w * r), int(512 * w * r), num_bottlenecks=int(3 * d), shortcut=True)

        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv_0(x)
        print(f"After conv_0: {x.shape}")
        x = self.conv_1(x)
        print(f"After conv_1: {x.shape}")
        x = self.c2f_2(x)
        print(f"After c2f_2: {x.shape}")
        x = self.conv_3(x)
        print(f"After conv_3: {x.shape}")
        out1 = self.c2f_4(x)
        print(f"After c2f_4: {out1.shape} <----- out1 shape")
        x = self.conv_5(out1)
        print(f"After conv_5: {x.shape}")
        out2 = self.c2f_6(x)
        print(f"After c2f_6: {out2.shape} <----- out2 shape")
        x = self.conv_7(out2)
        print(f"After conv_7: {x.shape}")
        x = self.c2f_8(x)
        print(f"After c2f_8: {x.shape}")
        x = self.sppf(x)
        print(f"After sppf: {x.shape}")
        return out1, out2, x


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)
        self.up = Upsample()
        self.c2f_1 = C2f(int(512 * w * (1+r)), int(512 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_2 = C2f(int(768 * w), int(256 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_3 = C2f(int(768 * w), int(512 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_4 = C2f(int(512 * w * (1+r)), int(512 * w * r), num_bottlenecks=int(3 * d), shortcut=False)
        self.cv_1 = Conv(int(256 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(int(512 * w), int(512 * w), kernel_size=3, stride=2, padding=1)
    
    def forward(self, x_res_1, x_res_2, x):
        print(f"Input shapes: x_res_1: {x_res_1.shape}, x_res_2: {x_res_2.shape}, x (res_1): {x.shape}")
        res_1 = x
        x = self.up(x)
        print(f"After upsample: {x.shape}")
        x = torch.cat((x, x_res_2), dim=1)
        print(f"After concatenation with x_res_2: {x.shape}")
        res_2 = self.c2f_1(x)
        print(f"After c2f_1: {res_2.shape} <-- res_2 shape")
        x = self.up(res_2)
        print(f"After upsample: {x.shape}")
        x = torch.cat((x, x_res_1), dim=1)
        print(f"After concatenation with x_res_1: {x.shape}")
        out_1 = self.c2f_2(x)
        print(f"After c2f_2: {out_1.shape} <----- out_1 shape")
        x = self.cv_1(out_1)
        print(f"After cv_1: {x.shape}")
        x = torch.cat((x, res_2), dim=1)
        print(f"After concatenation with res_2: {x.shape}")
        out_2 = self.c2f_3(x)
        print(f"After c2f_3: {out_2.shape} <----- out_2 shape")
        x = self.cv_2(out_2)
        print(f"After cv_2: {x.shape}")
        x = torch.cat((x, res_1), dim=1)
        print(f"After concatenation with res_1: {x.shape}")
        out_3 = self.c2f_4(x)
        print(f"After c2f_4: {out_3.shape} <----- out_3 shape")
        return out_1, out_2, out_3


class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)
    
    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        return x.view(b, 4, a)
    
class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=2):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch * 4
        self.nc = num_classes
        self.no = self.coordinates + self.nc
        self.stride = torch.zeros(3)
        d, w, r = yolo_params(version)

        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w * r), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1))
        ])

        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w * r), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1))
        ])

        self.dfl = DFL()

    def forward(self, x):
        print(f"Head input shapes: {[i.shape for i in x]}")
        for i in range(len(self.box)):
            box = self.box[i](x[i])
            clas = self.cls[i](x[i])
            x[i] = torch.cat((box, clas), dim=1)
                
        if self.training:
            return x
        
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, clas = x.split(split_size=(4 * self.ch, self.nc), dim=1)
        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, (b - a)), dim=1)
        return torch.cat(tensors=(box * strides, clas.sigmoid()), dim=1)
    
    def make_anchors(self, x, strides, offset=0.5):
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, __, h, w = x[i].shape
            sx = torch.arange(end=w, dtype=dtype, device=device) + offset
            sy = torch.arange(end=h, dtype=dtype, device=device) + offset
            sy, sx = torch.meshgrid(sy, sx)
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


class Yolo(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.backbone = Backbone(version)
        self.neck = Neck(version)
        self.head = Head(version)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))
