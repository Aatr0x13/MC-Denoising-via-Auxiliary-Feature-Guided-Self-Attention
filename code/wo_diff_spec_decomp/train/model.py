import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat


def print_model_structure(model):
    blank = ' '
    print('\t '+'-' * 95)
    print('\t ' + '|' + ' ' * 13 + 'weight name' + ' ' * 13 + '|' + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('\t ' + '-' * 95)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 35:
            key = key + (35 - len(key)) * blank
        else:
            key = key[:32] + '...'
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        else:
            shape = shape[:37] + '...'
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('\t ' + '| {} | {} | {} |'.format(key, shape, str_num))
    print('\t ' + '-' * 95)
    print('\t ' + 'Total number of parameters: ' + str(num_para))
    print('\t CUDA: ' + str(next(model.parameters()).is_cuda))
    print('\t ' + '-' * 95 + '\n')


def norm(norm_type, out_ch):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(out_ch, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(out_ch, affine=False)
    else:
        raise NotImplementedError('Normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for lrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('Activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# conv norm activation
def conv_block(in_ch, out_ch, kernel_size, stride=1, dilation=1, padding=0, padding_mode='zeros', norm_type=None,
               act_type='relu', groups=1, inplace=True):
    c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding,
                  padding_mode=padding_mode, groups=groups)
    n = norm(norm_type, out_ch) if norm_type else None
    a = act(act_type, inplace) if act_type else None
    return sequential(c, n, a)


class DiscriminatorVGG128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu'):
        super(DiscriminatorVGG128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, padding=1)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type, padding=1)
        # 64, 64
        conv2 = conv_block(base_nf, base_nf * 2, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type,
                           padding=1)
        conv3 = conv_block(base_nf * 2, base_nf * 2, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type,
                           padding=1)
        # 32, 128
        conv4 = conv_block(base_nf * 2, base_nf * 4, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type,
                           padding=1)
        conv5 = conv_block(base_nf * 4, base_nf * 4, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type,
                           padding=1)
        # 16, 256
        conv6 = conv_block(base_nf * 4, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type,
                           padding=1)
        conv7 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type,
                           padding=1)
        # 8, 512
        conv8 = conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type,
                           padding=1)
        conv9 = conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type, act_type=act_type,
                           padding=1)
        # 4, 512
        self.features = nn.Sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape((x.size(0), -1))
        x = self.classifier(x)
        return x


class AFGSA(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, bias=False):
        super(AFGSA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)

        self.conv_map = conv_block(ch*2, ch, kernel_size=1, act_type='relu')
        self.q_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, noisy, aux):
        n_aux = self.conv_map(torch.cat([noisy, aux], dim=1))
        b, c, h, w, block, halo, heads = *noisy.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0, 'feature map dimensions must be divisible by the block size'

        q = self.q_conv(n_aux)
        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q *= self.head_ch ** -0.5  # b*#blocks, flattened_query, c

        k = self.k_conv(n_aux)
        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = self.v_conv(noisy)
        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.q_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.k_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.v_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class TransformerBlock(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, checkpoint=True):
        super(TransformerBlock, self).__init__()
        self.checkpoint = checkpoint
        self.attention = AFGSA(ch, block_size=block_size, halo_size=halo_size, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            conv_block(ch, ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu'),
            conv_block(ch, ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu')
        )

    def forward(self, x):
        if self.checkpoint:
            noisy = x[0] + checkpoint(self.attention, x[0], x[1])
        else:
            noisy = x[0] + self.attention(x[0], x[1])
        noisy = noisy + self.feed_forward(noisy)
        return (noisy, x[1])


class AFGSANet(nn.Module):
    def __init__(self, in_ch, aux_in_ch, base_ch, num_sa=5, block_size=8, halo_size=3, num_heads=4, num_gcp=2):
        super(AFGSANet, self).__init__()
        assert num_gcp <= num_sa

        self.conv1 = conv_block(in_ch, 256, kernel_size=1, act_type='relu')
        self.conv3 = conv_block(in_ch, 256, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu')
        self.conv5 = conv_block(in_ch, 256, kernel_size=5, padding=2, padding_mode='reflect', act_type='relu')
        self.conv_map = conv_block(256*3, base_ch, kernel_size=1, act_type='relu')

        self.conv_a1 = conv_block(aux_in_ch, 256, kernel_size=1, act_type='relu')
        self.conv_a3 = conv_block(aux_in_ch, 256, kernel_size=3, padding=1, padding_mode='reflect', act_type='leakyrelu')
        self.conv_a5 = conv_block(aux_in_ch, 256, kernel_size=5, padding=2, padding_mode='reflect', act_type='leakyrelu')
        self.conv_aenc1 = conv_block(256*3, base_ch, kernel_size=1, act_type='leakyrelu')
        self.conv_aenc2 = conv_block(base_ch, base_ch, kernel_size=1, act_type='leakyrelu')

        transformer_blocks = []
        # to train on a RTX 3090, use gradient checkpoint for 3 Transformer blocks here (5 in total)
        for i in range(1, num_sa+1):
            if i <= (num_sa - num_gcp):
                transformer_blocks.append(TransformerBlock(base_ch, block_size=block_size, halo_size=halo_size,
                                                           num_heads=num_heads, checkpoint=False))
            else:
                transformer_blocks.append(TransformerBlock(base_ch, block_size=block_size, halo_size=halo_size,
                                                           num_heads=num_heads))
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.decoder = nn.Sequential(
            conv_block(base_ch, base_ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu'),
            conv_block(base_ch, base_ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu'),
            conv_block(base_ch, 3, kernel_size=3, padding=1, padding_mode='zeros', act_type=None)
        )

    def forward(self, x, aux):
        n1 = self.conv1(x)
        n3 = self.conv3(x)
        n5 = self.conv5(x)
        out = self.conv_map(torch.cat([n1, n3, n5], dim=1))

        a1 = self.conv_a1(aux)
        a3 = self.conv_a3(aux)
        a5 = self.conv_a5(aux)
        a = self.conv_aenc1(torch.cat([a1, a3, a5], dim=1))
        a = self.conv_aenc2(a)

        out = self.transformer_blocks([out, a])
        out = self.decoder(out[0])
        out += x
        return out


