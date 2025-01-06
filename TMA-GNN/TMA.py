import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from common.utils import norm_Adj, re_normalization
from prepareData import onehot_to_phase, generate_actphase, revise_unknown
import sys

sys.path.append("..")

def spline_quantile_loss(y_pred, y_true, mask_matrix, type, device):
    # y_true_1 = torch.unsqueeze(y_true, -1)
    labels = y_true
    y_true_1 = y_true.transpose(-1, -2)
    y_pred = y_pred.transpose(-1, -2)
    print('y_pred',y_pred.shape)
    print('y_true_1',y_true_1.shape)
    length = len(labels.shape)
    mask = torch.ones_like(labels)
    for idx, i in enumerate(mask_matrix):
        if type == 'train':
            # only compute observable node
            if i == 0 or i == 2:
                if length == 4:
                    mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                else:
                    mask[idx, :] = torch.zeros_like(mask[idx, :])
        else:
            if i == 0:
                if length == 4:
                    mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                else:
                    mask[idx, :] = torch.zeros_like(mask[idx, :])
    mask = mask.float()



    gamma = y_pred[:,:,:,:1]
    beta = y_pred[:,:,:,1:6]
    m = nn.Softmax(dim =3)
    delta = m(y_pred[:,:,:,6:11])

    b = beta - F.pad(beta, (1, 0))[:, :, :, :-1]
    d = F.pad(torch.cumsum(delta, dim=3), (1, 0))[:, :, :, :-1]  # d term for piecewise-linear functions

    value_knot = torch.add(F.pad(torch.cumsum(beta*delta, dim=3), (1, 0)),gamma)

    mask_1 = torch.where(value_knot >=
        y_true_1,torch.zeros(value_knot.shape).to(device),torch.ones(value_knot.shape).to(device))
    mask1 = mask_1[:, :, :, :-1]

    a_tilde_1 = (y_true_1-gamma+ torch.sum(mask1*b*d,3,keepdim=True))/ (1e-10+torch.sum(mask1*b,3,keepdim=True))
    a_tilde = torch.max(torch.min(a_tilde_1,torch.ones(a_tilde_1.shape).to(device)),torch.zeros(a_tilde_1.shape).to(device))

    coeff = (1.0 - torch.pow(d, 3)) / 3.0 - d - torch.pow(torch.max(a_tilde,d),2) + 2 * torch.max(a_tilde,d) * d
    crps = (2 * a_tilde - 1) * y_true_1 + (1 - 2 * a_tilde) * gamma + torch.sum(b * coeff,3,keepdim=True)

    loss = mask * crps
# trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0

    return loss.mean()

def maemis_loss(y_pred, labels, mask_matrix, type, device):
    # print(1,labels.shape)
    y_true = labels.transpose(-1, -2)
    # print(2222)
    print('y_true',y_true.shape)
    # print(y_true)
    print('y_pred',y_pred.shape)
    # print(y_true)

    length = len(labels.shape)
    pho = 0.05
    mask = torch.ones_like(labels)
    for idx, i in enumerate(mask_matrix):
        if type == 'train':
            # only compute observable node
            if i == 0 or i == 2:
                if length == 4:
                    mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                else:
                    mask[idx, :] = torch.zeros_like(mask[idx, :])
        else:
            if i == 0:
                if length == 4:
                    mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                else:
                    mask[idx, :] = torch.zeros_like(mask[idx, :])
    mask = mask.float()
    # print(y_pred.shape)
    # print('y_pred_2', y_pred.T[2].T.shape)
    loss0 = torch.abs(y_pred.T[2].T.unsqueeze(-1) - y_true)
    loss1 = torch.max(y_pred.T[0].T.unsqueeze(-1)-y_pred.T[1].T.unsqueeze(-1),torch.tensor([0.]).to(device))
    loss2 = torch.max(y_pred.T[1].T.unsqueeze(-1)-y_true,torch.tensor([0.]).to(device))*2/pho
    loss3 = torch.max(y_true-y_pred.T[0].T.unsqueeze(-1),torch.tensor([0.]).to(device))*2/pho
    loss = loss0+loss1+loss2+loss3
    loss = loss * mask
    loss[loss != loss] = 0
    # print(loss.shape)
    # print(loss)
    return loss.mean()



class EvidentialLossSumOfSquares(nn.Module):
    """The evidential loss function on a matrix.
    This class is implemented with slight modifications from the paper. The major
    change is in the regularizer parameter mentioned in the paper. The regularizer
    mentioned in the paper didnot give the required results, so we modified it
    with the KL divergence regularizer from the paper. In orderto overcome the problem
    that KL divergence are missing near zero so we add the minimum values to alpha,
    beta and lambda and compare distance with NIG(alpha=1.0, beta=0.1, lambda=1.0)
    This class only allows for rank-4 inputs for the output `targets`, and expectes
    `inputs` be of the form [mu, alpha, beta, lambda]
    alpha, beta and lambda needs to be positive values.
    """

    def __init__(self, debug=False, return_all=False):
        """Sets up loss function.
        Args:
          debug: When set to 'true' prints all the intermittent values
          return_all: When set to 'true' returns all loss values without taking average
        """
        super(EvidentialLossSumOfSquares, self).__init__()

        self.debug = debug
        self.return_all_values = return_all
        self.MAX_CLAMP_VALUE = 5.0  # Max you can go is 85 because exp(86) is nan  Now exp(5.0) is 143 which is max of a,b and l

    def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
        alpha_2 = torch.ones_like(mu1) * 1.0
        beta_2 = torch.ones_like(mu1) * 0.1
        lambda_2 = torch.ones_like(mu1) * 1.0

        t1 = 0.5 * (alpha_1 / beta_1) * ((mu1 - mu2) ** 2) * lambda_2
        # t1 = 0.5 * (alpha_1/beta_1) * (torch.abs(mu1 - mu2))  * lambda_2
        t2 = 0.5 * lambda_2 / lambda_1
        t3 = alpha_2 * torch.log(beta_1 / beta_2)
        t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
        t5 = (alpha_1 - alpha_2) * torch.digamma(alpha_1)
        t6 = -(beta_1 - beta_2) * (alpha_1 / beta_1)
        return (t1 + t2 - 0.5 + t3 + t4 + t5 + t6).to(self.device)

    def forward(self, inputs, targets, mask_matrix, type, DEVICE):
        self.device = DEVICE
        """ Implements the loss function
        Args:
          inputs: The output of the neural network. inputs has 4 dimension
            in the format [mu, alpha, beta, lambda]. Must be a tensor of
            floats
          targets: The expected output
        Returns:
          Based on the `return_all` it will return mean loss of batch or individual loss
        """
        assert torch.is_tensor(inputs)
        assert torch.is_tensor(targets)
        labels = targets
        #assert (inputs[:, 1, :] > 0).all()
        #assert (inputs[:, 2, :] > 0).all()
        #assert (inputs[:, 3, :] > 0).all()
        length = len(labels.shape)
        mask = torch.ones_like(labels)
        for idx, i in enumerate(mask_matrix):
            if type == 'train':
                # only compute observable node
                if i == 0 or i == 2:
                    if length == 4:
                        mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                    else:
                        mask[idx, :] = torch.zeros_like(mask[idx, :])
            else:
                if i == 0:
                    if length == 4:
                        mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                    else:
                        mask[idx, :] = torch.zeros_like(mask[idx, :])
        mask = mask.float()

        targets = targets.view(1, -1).to(DEVICE)
        y = inputs[:, :, 0, :].reshape(1, -1).to(DEVICE)
        a = torch.add(inputs[:, :, 1, :].reshape(1, -1), 1.0).to(DEVICE)
        b = torch.add(inputs[:, :, 2, :].reshape(1, -1), 0.3).to(DEVICE)
        l = torch.add(inputs[:, :, 3, :].reshape(1, -1), 1.0).to(DEVICE)

        if self.debug:
            print("a :", a)
            print("b :", b)
            print("l :", l)
        print(a.shape)
        # print(b.shape)
        # print(l.shape)
        J1 = torch.lgamma(a - 0.5).to(DEVICE)
        J2 = -torch.log(torch.tensor([4.0])).to(DEVICE)
        J3 = -torch.lgamma(a).to(DEVICE)
        J4 = -torch.log(l).to(DEVICE)
        J5 = -0.5 * torch.log(b).to(DEVICE)
        J6 = torch.log(2 * b * (1 + l) + (2 * a - 1) * l * (y - targets) ** 2).to(DEVICE)
        # print('让我看看')
        # print(J6.device)
        if self.debug:
            print("lgama(a - 0.5) :", J1)
            print("log(4):", J2)
            print("lgama(a) :", J3)
            print("log(l) :", J4)
            print("log( ---- ) :", J6)

        J = J1 + J2 + J3 + J4 + J5 + J6
        # Kl_divergence = torch.abs(y - targets) * (2*a + l)/b ######## ?????
        # Kl_divergence = ((y - targets)**2) * (2*a + l)
        # Kl_divergence = torch.abs(y - targets) * (2*a + l)
        # Kl_divergence = 0.0
        # Kl_divergence = (torch.abs(y - targets) * (a-1) *  l)/b
        Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)

        if self.debug:
            print("KL ", Kl_divergence.data.numpy())
        loss = torch.exp(J) + Kl_divergence
        # print('loss',loss)
        # print(loss.shape)
        # print('mask',mask)
        # print(mask.shape)
        mask = mask.view(1, -1).to(DEVICE)
        loss = loss * mask
        if self.debug:
            print("loss :", loss.mean())

        if self.return_all_values:
            ret_loss = loss
        else:
            ret_loss = loss.mean()
        # if torch.isnan(ret_loss):
        #  ret_loss.item() = self.prev_loss + 10
        # else:
        #  self.prev_loss = ret_loss.item()

        return ret_loss



class EvidentialLoss_encoder(nn.Module):
    """The evidential loss function on a matrix.
    This class is implemented with slight modifications from the paper. The major
    change is in the regularizer parameter mentioned in the paper. The regularizer
    mentioned in the paper didnot give the required results, so we modified it
    with the KL divergence regularizer from the paper. In orderto overcome the problem
    that KL divergence are missing near zero so we add the minimum values to alpha,
    beta and lambda and compare distance with NIG(alpha=1.0, beta=0.1, lambda=1.0)
    This class only allows for rank-4 inputs for the output `targets`, and expectes
    `inputs` be of the form [mu, alpha, beta, lambda]
    alpha, beta and lambda needs to be positive values.
    """

    def __init__(self, debug=False, return_all=False):
        """Sets up loss function.
        Args:
          debug: When set to 'true' prints all the intermittent values
          return_all: When set to 'true' returns all loss values without taking average
        """
        super(EvidentialLoss_encoder, self).__init__()

        self.debug = debug
        self.return_all_values = return_all
        self.MAX_CLAMP_VALUE = 5.0  # Max you can go is 85 because exp(86) is nan  Now exp(5.0) is 143 which is max of a,b and l

    def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
        alpha_2 = torch.ones_like(mu1) * 1.0
        beta_2 = torch.ones_like(mu1) * 0.1
        lambda_2 = torch.ones_like(mu1) * 1.0

        t1 = 0.5 * (alpha_1 / beta_1) * ((mu1 - mu2) ** 2) * lambda_2
        # t1 = 0.5 * (alpha_1/beta_1) * (torch.abs(mu1 - mu2))  * lambda_2
        t2 = 0.5 * lambda_2 / lambda_1
        t3 = alpha_2 * torch.log(beta_1 / beta_2)
        t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
        t5 = (alpha_1 - alpha_2) * torch.digamma(alpha_1)
        t6 = -(beta_1 - beta_2) * (alpha_1 / beta_1)
        return (t1 + t2 - 0.5 + t3 + t4 + t5 + t6).to(self.device)

    def forward(self, inputs, targets, mask_matrix, type, DEVICE):
        self.device = DEVICE
        """ Implements the loss function
        Args:
          inputs: The output of the neural network. inputs has 4 dimension
            in the format [mu, alpha, beta, lambda]. Must be a tensor of
            floats
          targets: The expected output
        Returns:
          Based on the `return_all` it will return mean loss of batch or individual loss
        """
        assert torch.is_tensor(inputs)
        assert torch.is_tensor(targets)
        labels = targets
        #assert (inputs[:, 1, :] > 0).all()
        #assert (inputs[:, 2, :] > 0).all()
        #assert (inputs[:, 3, :] > 0).all()
        length = len(labels.shape)
        mask = torch.ones_like(labels)

        for idx, i in enumerate(mask_matrix):
            if type == 'train':
                # only compute observable node
                if i == 0 or i == 2:
                    if length == 5:
                        mask[:, idx, :, :, :] = torch.zeros_like(mask[:, idx, :, :, :])
                    else:
                        mask[idx, :] = torch.zeros_like(mask[idx, :])
            else:
                if i == 0:
                    if length == 5:
                        mask[:, idx, :, :, :] = torch.zeros_like(mask[:, idx, :, :, :])
                    else:
                        mask[idx, :] = torch.zeros_like(mask[idx, :])
        mask = mask.float()
        # print('1', inputs[:, :, 1, :, :].reshape(1, -1))
        # print('2', torch.add(inputs[:, :, 1, :, :].reshape(1, -1), 1.0))
        targets = targets.reshape(1, -1).to(DEVICE)
        y = inputs[:, :, 0, :, :].reshape(1, -1).to(DEVICE)
        a = torch.add(inputs[:, :, 1, :, :].reshape(1, -1), 1.0).to(DEVICE)
        b = torch.add(inputs[:, :, 2, :, :].reshape(1, -1), 0.3).to(DEVICE)
        l = torch.add(inputs[:, :, 3, :, :].reshape(1, -1), 1.0).to(DEVICE)
        # print('3', a)
        if self.debug:
            print("a :", a)
            print("b :", b)
            print("l :", l)

        J1 = torch.lgamma(a - 0.5).to(DEVICE)
        J2 = -torch.log(torch.tensor([4.0])).to(DEVICE)
        J3 = -torch.lgamma(a).to(DEVICE)
        J4 = -torch.log(l).to(DEVICE)
        J5 = -0.5 * torch.log(b).to(DEVICE)
        J6 = torch.log(2 * b * (1 + l) + (2 * a - 1) * l * (y - targets) ** 2).to(DEVICE)
        # print('让我看看')
        # print(J6.device)
        if self.debug:
            print("lgama(a - 0.5) :", J1)
            print("log(4):", J2)
            print("lgama(a) :", J3)
            print("log(l) :", J4)
            print("log( ---- ) :", J6)

        J = J1 + J2 + J3 + J4 + J5 + J6

        # Kl_divergence = torch.abs(y - targets) * (2*a + l)/b ######## ?????
        # Kl_divergence = ((y - targets)**2) * (2*a + l)
        # Kl_divergence = torch.abs(y - targets) * (2*a + l)
        # Kl_divergence = 0.0
        # Kl_divergence = (torch.abs(y - targets) * (a-1) *  l)/b
        Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)

        if self.debug:
            print("KL ", Kl_divergence.data.numpy())
        loss = torch.exp(J) + Kl_divergence

        # print(loss)
        # print('loss',loss)
        # print(loss.shape)
        # print('mask',mask)
        # print(mask.shape)
        mask = mask.reshape(1, -1).to(DEVICE)
        loss = loss * mask
        # print(loss)
        if self.debug:
            print("loss :", loss.mean())

        if self.return_all_values:
            ret_loss = loss
        else:
            ret_loss = torch.mean(loss[~torch.isnan(loss)])

        # if torch.isnan(ret_loss):
        #  ret_loss.item() = self.prev_loss + 10
        # else:
        #  self.prev_loss = ret_loss.item()

        return ret_loss


def clones(module, N):
    '''
    produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 1 means reachable; 0 means unreachable
    return torch.from_numpy(subsequent_mask) == 0


# class spatialGCN(nn.Module):
#     def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
#         super(spatialGCN, self).__init__()
#         self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.Theta = nn.Linear(in_channels, out_channels, bias=False)

#     def forward(self, x):
#         '''
#         spatial graph convolution operation
#         x: (batch_size, N, T, F_in)
#         :return: (batch_size, N, T, F_out)
#         '''
#         batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

#         x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

#         return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)


class Temporal_Attention_layer(nn.Module):
    '''
    compute temporal attention scores
    '''

    def __init__(self, dropout=.0):
        super(Temporal_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 1, 2, 3).reshape((-1, num_of_vertices, in_channels))  # (b*N,T,f_in)

        # (b*N,T,f_in)(b*N,f_in,N)=(b*N, T, T)
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)

        # the sum of each row is 1; (b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))

        return score.reshape((batch_size, num_of_vertices, num_of_timesteps, num_of_timesteps))


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)

        # the sum of each row is 1; (b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


# class spatialAttentionGCN(nn.Module):
#     def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
#         super(spatialAttentionGCN, self).__init__()
#         self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.Theta = nn.Linear(in_channels, out_channels, bias=False)
#         self.SAt = Spatial_Attention_layer(dropout=dropout)

#     def forward(self, x):
#         '''
#         spatial graph convolution operation
#         x: (batch_size, N, T, F_in)
#         :return: (batch_size, N, T, F_out)
#         '''

#         batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

#         spatial_attention = self.SAt(x)  # (batch, T, N, N)

#         x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

#         spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)
#         # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)
#         return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class temporalAttentionScaledGCN(nn.Module):
    def __init__(self, DEVICE, sym_norm_Adj_matrix, mask_matrix, in_channels, out_channels, dropout=.0):
        super(temporalAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.mask_matrix = mask_matrix
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Temporal_Attention_layer(dropout=dropout)
        self.DEVICE = DEVICE

    def forward(self, x, phaseAct_matrix):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x) / math.sqrt(in_channels)
        print(spatial_attention.shape)


class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, DEVICE, sym_norm_Adj_matrix, mask_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.mask_matrix = mask_matrix
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)
        self.DEVICE = DEVICE
        self.Lin = nn.Linear(30, 1, bias=False)

    def forward(self, x, phaseAct_matrix):
        '''
        spatial graph convolution operation,including imputation
        :param x: (batch_size, N, T, F_in) b t n f
        :param phaseAct_matrix: (B, T, N, N)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        for t in range(1, num_of_timesteps):
            x_ = x[:, :, :num_of_timesteps - 1, :].permute(0, 2, 1, 3)
            x_ = x_.reshape((-1, num_of_vertices, in_channels))  # (b*T,N,f_in)
            x__ = x[:, :, 1:num_of_timesteps, :].permute(0, 2, 1, 3)
            x__ = x__.reshape((-1, num_of_vertices, in_channels))  # (b*T,N,f_in)
            score = torch.matmul(x_, x__.transpose(1, 2)) / math.sqrt(in_channels)
            score = nn.Dropout(p=.0)(F.softmax(score, dim=-1))
            score = score.reshape((batch_size, 29, num_of_vertices, num_of_vertices))

            # 都求
            # 首先计算两个张量的点积，以计算注意力分数
            attention_scores = torch.matmul(x.permute(0, 1, 2, 3), x.permute(0, 1, 2, 3).transpose(-1, -2)) / math.sqrt(
                in_channels)
            # 然后对分数进行 softmax，以获取注意力权重
            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
            output = self.Lin(attention_weights).permute(0, 2, 1, 3)

            # scaled self attention: (B, T, N, N)
            spatial_attention = self.SAt(x) / math.sqrt(in_channels)
            # (B,T,N,N)-permute->(B,T,N,N)
            # sat_act = (phaseAct_matrix[:, 1:] * spatial_attention[:, 1:] * output[:, 1:]).permute(0, 1, 3, 2)
            sat_act = (phaseAct_matrix[:, 1:] * score * output[:, 1:]).permute(0, 1, 3, 2)
            # (B,T,N,N)(B,T,N,F)->(B,T,N,F)-permute->(B,N,F,T)
            x_predict = torch.matmul(sat_act, x.permute(0, 2, 1, 3)[:, :num_of_timesteps - 1]).permute(0, 2, 3, 1)
            # (B,N,F,T)-permute->(B,N,T,F)
            x = (revise_unknown(x.permute(0, 1, 3, 2), x_predict, self.mask_matrix).to(self.DEVICE)).permute(0, 1, 3, 2)

        spatial_attention = (self.SAt(x) / math.sqrt(in_channels)).reshape(
            (-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        phaseAct_adj = (phaseAct_matrix * self.sym_norm_Adj_matrix).reshape(-1, num_of_vertices, num_of_vertices)

        satAct_adj = phaseAct_adj.mul(spatial_attention)
        # (B,N,T,F)->permute->(B,T,N,F)-reshape->(B*T,N,F)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))

        # (B*T,N,F_in)->(B*T,N, F_out)-reshape->(B,T,N,F_out)->(B,N,T,F_out)
        return F.relu(self.Theta(torch.matmul(satAct_adj.permute(0, 2, 1), x)).reshape(
            (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        # 在模型中定义一个常量，.step时不会被更新
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in(64))
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            # (batch_size, N, T, F_in) + (1,1,T,d_model)
            x = x + self.pe[:, :, self.lookup_index, :]
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''

    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, layernum, phaseAct_matrix, x, sublayer):
        '''
        x: (batch, N, T, d_model)
        sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            if layernum != 2:
                return x + self.dropout(sublayer(self.norm(x)))
            else:
                gcn = sublayer(self.norm(x), phaseAct_matrix)
                return x + self.dropout(gcn)
        if self.residual_connection and (not self.use_LayerNorm):
            if layernum != 2:
                return x + self.dropout(sublayer(x))
            else:
                gcn = sublayer(x, phaseAct_matrix)
                return x + self.dropout(gcn)
        if (not self.residual_connection) and self.use_LayerNorm:
            if layernum != 2:
                return self.dropout(sublayer(self.norm(x)))
            else:
                gcn = sublayer(self.norm(x), phaseAct_matrix)
                return self.dropout(gcn)


class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phaseAct_matrix):
        '''
        :param x: (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        gcn = self.gcn(x, phaseAct_matrix)
        return self.dropout(F.relu(gcn))


def attention(query, key, value, mask=None, dropout=None):
    '''
    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / \
             math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        # -1e9 means attention scores=0
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


# key causal; query causal;
class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):
    def __init__(self, nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_mhalf * points_per_mhalf

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.h_length > 0:
                query_h, key_h = [
                    l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2) for
                    l, x in zip(self.conv1Ds_aware_temporal_context,
                                (query[:, :, :self.h_length, :], key[:, :, :self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [
                l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                    -1).permute(0, 3, 1, 4, 2) for l, x
                in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[
                    :, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.h_length, :].permute(
                    0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(
                    0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 1d conv on query, 1d conv on key
class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = (kernel_size - 1) // 2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_mhalf * points_per_mhalf

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        query=key=value
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.h_length > 0:
                # l:Conv2d
                query_h, key_h = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context,
                                    (query[:, :, :self.h_length, :], key[:, :, : self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)
            # (batch, N, h, T, d_k)
            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(
                0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous(
            ).view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.h_length, :].permute(
                    0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


# query: causal conv; key 1d conv
class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_mhalf, points_per_mhalf, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1) // 2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                              padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                            padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_mhalf * points_per_mhalf

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(query[:, :, : self.h_length, :].permute(
                    0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                   -1).permute(0, 3, 1, 4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, : self.h_length, :].permute(
                    0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h,
                                                                                                       self.d_k, N,
                                                                                                       -1).permute(0, 3,
                                                                                                                   1, 4,
                                                                                                                   2)

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, : self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        # (batch, N, T1, d_model)
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    def __init__(self, PAct, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.PAct = PAct
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.generator1 = nn.Linear(64, 3*3)
        self.generator2 = nn.Linear(64, 11*3)
        self.refill = nn.Linear(64, 3*3)
        self.lin_prob = nn.Linear(1,3)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, src, trg, _mean, _std):
        '''
        :param src:  (batch_size, N, T_in, F_in)
        :param trg: (batch, N, T_out, F_out)
        :param _mean: (1, 1, F, 1)
        :param _std: (1, 1, F, 1)
        '''
        encoder_output, encoder_refill, refill_prob = self.encode(src, _mean, _std)  # (batch_size, N, T_in, d_model)

        return self.decode(trg, encoder_output), encoder_refill, refill_prob

    def encode(self, src, _mean, _std):
        '''
        :param src: (batch_size, N, T_in, F_in)
        :param _mean: (1, 1, F, 1)
        :param _std: (1, 1, F, 1)
        : returns: encoder_output:(B, N, T, F)
        '''
        phaseAct_matrix = self.PAct(src, _mean, _std)  # (b,T,N,N)
        phaseAct_matrix = torch.from_numpy(phaseAct_matrix).type(torch.FloatTensor).to(self.DEVICE)
        encoder_output = self.encoder(self.src_embed(src), phaseAct_matrix)
        encoder_refill = self.prediction_generator(encoder_output)
        refill_prob = self.refill(encoder_output)
        refill_prob = refill_prob.view(-1, 80, 30, 3, 3)
        encoder_refill = encoder_refill.unsqueeze(-1)
        print(encoder_refill.shape)
        print(refill_prob.shape)
        refill_prob = torch.cat((encoder_refill, refill_prob), dim= -1)
        encoder_refill = encoder_refill.squeeze(-1)
        return encoder_output, encoder_refill, refill_prob

    def decode(self, trg, encoder_output):
        '''
        :param trg:(batch_size, N, T, F(3))
        :param encoder_output: (B, N, T, F)
        :return: (B, N, T, F)
        '''
        batch_size, N, T, _ = trg.shape
        phaseAct_matrix = torch.ones(batch_size, T, N, N).type(torch.FloatTensor).to(self.DEVICE)
        h = self.trg_embed(trg)
        output = self.decoder(h, encoder_output, phaseAct_matrix)
        output1 = self.prediction_generator(output)
        # print(output.permute(0,1,3,2).shape)

        projected = self.generator1(output)
        print('projected', projected.shape)
        out2 = projected.view(-1, 80, 3, 3)
        # out = self.lin_prob(output.permute(0,1,3,2))
        # print(out.shape)
        output1[output1 < 0] = 0
        out2[out2 < 0] = 0
        out2 = torch.cat((output1, out2), dim=2)
        return output1, out2


class EncoderLayer(nn.Module):
    def __init__(self, size, sat_act, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.sat_act = sat_act
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(
                size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x, phaseAct_matrix):
        '''
        :param x: (B, N, T_in, F_in)
        :param phaseAct_matrix: (B, T, N, N)
        :return: (B, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](0, phaseAct_matrix, x,
                                 lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](2, phaseAct_matrix, x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x, phaseAct_matrix)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, phaseAct_matrix):
        '''
        :param x: (B, N, T_in, F_in)
        :param phaseAct_matrix: (B, T, N, N)
        :return: (B, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x, phaseAct_matrix)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory, phaseAct_matrix):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            # self.self_attn: captures the correlation in the decoder sequence
            x = self.sublayer[0](0, phaseAct_matrix, x,
                                 lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False,
                                                          key_multi_segment=False))  # output: (batch, N, T', d_model)
            # self.src_attn: capture the correlations between the decoder sequence (queries) and the encoder output sequence(keys)
            x = self.sublayer[1](1, phaseAct_matrix, x, lambda x: self.src_attn(x, m, m, query_multi_segment=False,
                                                                                key_multi_segment=True))  # output: (batch, N, T', d_model)
            # output:  (batch, N, T', d_model)
            return self.sublayer[2](2, phaseAct_matrix, x, self.feed_forward_gcn)
        else:
            # output: (batch, N, T', d_model)
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)
            # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)
            # output:  (batch, N, T', d_model)
            return self.feed_forward_gcn(x, phaseAct_matrix)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, phaseAct_matrix):
        '''
        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory, phaseAct_matrix)
        return self.norm(x)


def search_index(max_len, num_of_depend, num_for_predict, points_per_mhalf):
    '''
    :param max_len: int, length of all encoder input
    :param num_of_depend: int,
    :param num_for_predict: int, the number of points will be predicted for each sample
    :param points_per_mhalf: int, number of points per hour, depends on data
    :return: list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_mhalf * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


class Phase_Act_layer(nn.Module):
    def __init__(self, adj_mx, adj_phase):
        super(Phase_Act_layer, self).__init__()
        self.adj_mx = adj_mx
        # (N,N)
        self.adj_phase = adj_phase

    def forward(self, x, _mean, _std):
        '''
        :param x:(b,N,T,F)
        :param _mean:(1,1,F(11),1)
        :param _std:(1,1,F(11),1)
        '''
        _, N, _, _ = x.shape
        x_renor = re_normalization(x.cpu().numpy().transpose(0, 1, 3, 2), _mean, _std)[:, :, 3:]
        x_renor_ = np.where((np.abs(x_renor - 1.) >= 1e-6), 0, 1)
        # (B,T,N,2)
        onehot2phase = onehot_to_phase(x_renor_)
        # (B,T,N,N)
        # compute phase_act matrix of each time according to adj_phase,x_next_phase
        phase_matrix = generate_actphase(onehot2phase, self.adj_mx, self.adj_phase)
        # add self_loop
        phase_matrix = phase_matrix + np.eye(N)
        return phase_matrix


def make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, adj_phase, mask_matrix,
               nb_head,
               num_of_mhalf, points_per_mhalf, num_for_predict, len_input, dropout=.0, aware_temporal_context=True,
               SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):
    c = copy.deepcopy

    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)

    num_of_vertices = norm_Adj_matrix.shape[0]

    src_dense = nn.Linear(encoder_input_size, d_model)
    PAct = Phase_Act_layer(adj_mx, adj_phase)

    sat_act = PositionWiseGCNFeedForward(
        spatialAttentionScaledGCN(DEVICE, norm_Adj_matrix, mask_matrix, d_model, d_model), dropout=dropout)

    position_wise_gcn = PositionWiseGCNFeedForward(
        spatialAttentionScaledGCN(DEVICE, norm_Adj_matrix, mask_matrix, d_model, d_model), dropout=dropout)

    # target input projection
    trg_dense = nn.Linear(decoder_input_size, d_model)

    # encoder temporal position embedding
    max_len = num_of_mhalf * points_per_mhalf

    h_index = search_index(max_len, num_of_mhalf, len_input, points_per_mhalf)
    en_lookup_index = h_index

    print('TemporalPositionalEncoding max_len:', max_len)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    if aware_temporal_context:  # employ temporal trend-aware attention
        attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_mhalf, points_per_mhalf,
                                                                kernel_size,
                                                                dropout=dropout)  # encoder的trend-aware attention用一维卷积
        attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_mhalf, points_per_mhalf,
                                                               kernel_size, dropout=dropout)
        att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_mhalf, points_per_mhalf,
                                                             kernel_size,
                                                             dropout=dropout)  # decoder的trend-aware attention用因果卷积
    else:  # employ traditional self attention
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len,
                                                              en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout,
                                                     GCN(norm_Adj_matrix, d_model, d_model),
                                                     smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout,
                                                     GCN(norm_Adj_matrix, d_model, d_model),
                                                     smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and (TE):
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len,
                                                              en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)

    encoderLayer = EncoderLayer(d_model, c(sat_act), attn_ss, c(position_wise_gcn), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    decoder = Decoder(decoderLayer, num_layers)

    generator = nn.Linear(d_model, decoder_input_size)

    model = EncoderDecoder(PAct,
                           encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE)
    # param init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
