from torch import nn

def get_net():
    net = nn.Sequential(
        nn.Conv2d(3,16,kernel_size=5)
    )
    return net
