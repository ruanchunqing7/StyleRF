import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random

def calc_mean_std(x, eps=1e-8):
        """
        calculating channel-wise instance mean and standard variance
        x: shape of (N,C,*)
        """
        mean = torch.mean(x.flatten(2), dim=-1, keepdim=True) # size of (N, C, 1)
        std = torch.std(x.flatten(2), dim=-1, keepdim=True) + eps # size of (N, C, 1)
        
        return mean, std

def cal_adain_style_loss(x, y):
    """
    style loss in one layer

    Args:
        x, y: feature maps of size [N, C, H, W]
    """
    x_mean, x_std = calc_mean_std(x)
    y_mean, y_std = calc_mean_std(y)

    return nn.functional.mse_loss(x_mean, y_mean) \
         + nn.functional.mse_loss(x_std, y_std)

def cal_mse_content_loss(x, y):
    return nn.functional.mse_loss(x, y)

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class LearnableIN(nn.Module):
    '''
    Input: (N, C, L) or (C, L)
    '''
    def __init__(self, dim=256):
        super().__init__()
        self.IN = torch.nn.InstanceNorm1d(dim, momentum=1e-4, track_running_stats=True)

    def forward(self, x):
        if x.size()[-1] <= 1:
            return x
        return self.IN(x)

class SimpleLinearStylizer(nn.Module):
    def __init__(self, input_dim=256, embed_dim=32, n_layers=3 ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.IN = LearnableIN(input_dim)

        self.q_embed = nn.Conv1d(input_dim, embed_dim, 1)
        self.k_embed = nn.Conv1d(input_dim, embed_dim, 1)
        self.v_embed = nn.Conv1d(input_dim, embed_dim, 1)

        self.unzipper = nn.Conv1d(embed_dim, input_dim, 1, bias=0)

        s_net = []
        for i in range(n_layers - 1):
            out_dim = max(embed_dim, input_dim // 2)
            s_net.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, out_dim, 1),
                    nn.ReLU(inplace=True),
                )
            )
            input_dim = out_dim
        s_net.append(nn.Conv1d(input_dim, embed_dim, 1))
        self.s_net = nn.Sequential(*s_net)

        self.s_fc = nn.Linear(embed_dim ** 2, embed_dim ** 2)

    def _vectorized_covariance(self, x):
        cov = torch.bmm(x, x.transpose(2, 1)) / x.size(-1)
        cov = cov.flatten(1)
        return cov

    def get_content_matrix(self, c):
        '''
        Args:
            c: content feature [N,input_dim,S]
        Return:
            mat: [N,S,embed_dim,embed_dim]
        '''
        normalized_c = self.IN(c)
        # normalized_c = torch.nn.functional.instance_norm(c)
        q_embed = self.q_embed(normalized_c)
        k_embed = self.k_embed(normalized_c)
        
        c_cov = q_embed.transpose(1,2).unsqueeze(3) * k_embed.transpose(1,2).unsqueeze(2) # [N,S,embed_dim,embed_dim]
        attn = torch.softmax(c_cov, -1) # [N,S,embed_dim,embed_dim]

        return attn, normalized_c

    def get_style_mean_std_matrix(self, s):
        '''
        Args:
            s: style feature [N,input_dim,S] [1, 256, 4096]

        Return:
            mat: [N,embed_dim,embed_dim]
        '''
        s_mean = s.mean(-1, keepdim=True)
        # [1, 256, 1]
        s_std = s.std(-1, keepdim=True)
        # [1, 256, 1]
        s = s - s_mean

        s_embed = self.s_net(s)
        s_cov = self._vectorized_covariance(s_embed)
        s_mat = self.s_fc(s_cov)
        s_mat = s_mat.reshape(-1, self.embed_dim, self.embed_dim)
        # [1, 32, 32]
        return s_mean, s_std, s_mat

    def transform_content_3D(self, c):
        '''
        Args:
            c: content feature [N,input_dim,S]
        Return:
            transformed_c: [N,embed_dim,S]
        '''
        attn, normalized_c = self.get_content_matrix(c) # [N,S,embed_dim,embed_dim]
        c = self.v_embed(normalized_c) # [N,embed_dim,S]
        c = c.transpose(1,2).unsqueeze(3) # [N,S,embed_dim,1]
        c = torch.matmul(attn, c).squeeze(3) # [N,S,embed_dim]

        return c.transpose(1,2)

    def transfer_style_2D(self, s_mean_std_mat, c, acc_map):
        '''
        Agrs:
            c: content feature map after volume rendering [N,embed_dim,S] [1, 32, 2048]
            s_mat: style matrix [N,embed_dim,embed_dim] [1, 32, 32]
            acc_map: [S] [2048]
            
            s_mean = [N,input_dim,1] [1, 256, 1]
            s_std = [N,input_dim,1] [1, 256, 1]
        '''
        s_mean, s_std, s_mat = s_mean_std_mat

        cs = torch.bmm(s_mat, c) # [N,embed_dim,S]
        # [1, 32, 2048]
        cs = self.unzipper(cs) # [N,input_dim,S]
        # [1, 256, 2048]
        cs = cs * s_std + s_mean * acc_map[None,None,...]
        # [1, 256, 2048]
        return cs

    def add_channel(self, c):
        c = self.unzipper(c)  # [N,input_dim,S]
        return c

class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

# class AdaAttN(nn.Module):
#     """ Attention-weighted AdaIN (Liu et al., ICCV 21) """
#
#     def __init__(self, qk_dim, v_dim):
#         """
#         Args:
#             qk_dim (int): query and key size.
#             v_dim (int): value size.
#         """
#         super(AdaAttN, self).__init__()
#
#         self.q_embed = nn.Conv1d(qk_dim, qk_dim, 1)
#         self.k_embed = nn.Conv1d(qk_dim, qk_dim, 1)
#         self.s_embed = nn.Conv1d(v_dim, v_dim, 1)
#
#     def forward(self, q, k):
#         """
#         Args:
#             q (float tensor, (bs, qk, *)): query (content) features.
#             k (float tensor, (bs, qk, *)): key (style) features.
#             c (float tensor, (bs, v, *)): content value features.
#             s (float tensor, (bs, v, *)): style value features.
#
#         Returns:
#             cs (float tensor, (bs, v, *)): stylized content features.
#         """
#         c, s = q, k
#
#         shape = c.shape
#         q, k = q.flatten(2), k.flatten(2)
#         c, s = c.flatten(2), s.flatten(2)
#
#         # QKV attention with projected content and style features
#         q = self.q_embed(F.instance_norm(q)).transpose(2, 1)    # (bs, n, qk)
#         k = self.k_embed(F.instance_norm(k))                    # (bs, qk, m)
#         s = self.s_embed(s).transpose(2, 1)                     # (bs, m, v)
#         attn = F.softmax(torch.bmm(q, k), -1)                   # (bs, n, m)
#
#         # attention-weighted channel-wise statistics
#         mean = torch.bmm(attn, s)                               # (bs, n, v)
#         var = F.relu(torch.bmm(attn, s ** 2) - mean ** 2)       # (bs, n, v)
#         mean = mean.transpose(2, 1)                             # (bs, v, n)
#         std = torch.sqrt(var).transpose(2, 1)                   # (bs, v, n)
#
#         cs = F.instance_norm(c) * std + mean                    # (bs, v, n)
#         cs = cs.reshape(shape)
#         return cs

class AdaAttN_new_IN(nn.Module):
    """ Attention-weighted AdaIN (Liu et al., ICCV 21) """

    def __init__(self, qk_dim, v_dim):
        """
        Args:
            qk_dim (int): query and key size.
            v_dim (int): value size.
        """
        super(AdaAttN_new_IN, self).__init__()

        self.q_embed = nn.Conv1d(qk_dim, qk_dim, 1)
        self.k_embed = nn.Conv1d(qk_dim, qk_dim, 1)
        self.s_embed = nn.Conv1d(v_dim, v_dim, 1)
        self.IN = LearnableIN(qk_dim)

    def forward(self, q, k):
        """
        Args:
            q (float tensor, (bs, qk, *)): query (content) features.
            k (float tensor, (bs, qk, *)): key (style) features.
            c (float tensor, (bs, v, *)): content value features.
            s (float tensor, (bs, v, *)): style value features.

        Returns:
            cs (float tensor, (bs, v, *)): stylized content features.
        """
        c, s = q, k

        shape = c.shape
        q, k = q.flatten(2), k.flatten(2)
        c, s = c.flatten(2), s.flatten(2)

        # QKV attention with projected content and style features
        q = self.q_embed(self.IN(q)).transpose(2, 1)    # (bs, n, qk)
        k = self.k_embed(F.instance_norm(k))                    # (bs, qk, m)
        s = self.s_embed(s).transpose(2, 1)                     # (bs, m, v)
        attn = F.softmax(torch.bmm(q, k), -1)                   # (bs, n, m)
        
        # attention-weighted channel-wise statistics
        mean = torch.bmm(attn, s)                               # (bs, n, v)
        var = F.relu(torch.bmm(attn, s ** 2) - mean ** 2)       # (bs, n, v)
        mean = mean.transpose(2, 1)                             # (bs, v, n)
        std = torch.sqrt(var).transpose(2, 1)                   # (bs, v, n)
        
        cs = self.IN(c) * std + mean                    # (bs, v, n)
        cs = cs.reshape(shape)
        return cs

class AdaAttN_woin(nn.Module):
    """ Attention-weighted AdaIN (Liu et al., ICCV 21) """

    def __init__(self, qk_dim, v_dim):
        """
        Args:
            qk_dim (int): query and key size.
            v_dim (int): value size.
        """
        super().__init__()

        self.q_embed = nn.Conv1d(qk_dim, qk_dim, 1)
        self.k_embed = nn.Conv1d(qk_dim, qk_dim, 1)
        self.s_embed = nn.Conv1d(v_dim, v_dim, 1)

    def forward(self, q, k):
        """
        Args:
            q (float tensor, (bs, qk, *)): query (content) features.
            k (float tensor, (bs, qk, *)): key (style) features.
            c (float tensor, (bs, v, *)): content value features.
            s (float tensor, (bs, v, *)): style value features.

        Returns:
            cs (float tensor, (bs, v, *)): stylized content features.
        """
        c, s = q, k

        shape = c.shape
        q, k = q.flatten(2), k.flatten(2)
        c, s = c.flatten(2), s.flatten(2)

        # QKV attention with projected content and style features
        q = self.q_embed(q).transpose(2, 1)    # (bs, n, qk)
        k = self.k_embed(k)                    # (bs, qk, m)
        s = self.s_embed(s).transpose(2, 1)                     # (bs, m, v)
        attn = F.softmax(torch.bmm(q, k), -1)                   # (bs, n, m)
        
        # attention-weighted channel-wise statistics
        mean = torch.bmm(attn, s)                               # (bs, n, v)
        var = F.relu(torch.bmm(attn, s ** 2) - mean ** 2)       # (bs, n, v)
        mean = mean.transpose(2, 1)                             # (bs, v, n)
        std = torch.sqrt(var).transpose(2, 1)                   # (bs, v, n)
        
        cs = c * std + mean                    # (bs, v, n)
        cs = cs.reshape(shape)
        return cs