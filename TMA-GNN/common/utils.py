import os
import numpy as np
import torch
import torch.utils.data
from scipy.sparse.linalg import eigs
from common.metrics import masked_mape_test, masked_mae_test, masked_rmse_test
# from metrics import masked_mape_test, masked_mae_test, masked_rmse_test
import pickle
from time import time


# def calculate_confidence_intervals(data, confidence_level=0.95):
#     B, N, X, F = data.shape
#     print(data.shape)
#     # data = data.transpose(0, 1, 3, 2)
#     data = data.permute(0, 1, 3, 2)
#     data = data.reshape(B * N * F, X)
#     mean = torch.mean(data, axis=1)
#     std_dev = torch.std(data, axis=1)
#     z_score = 1.96  # 对应于95%置信水平的z值
#
#     margin_of_error = z_score * (std_dev / np.sqrt(X))
#     lower_bound = torch.max(mean - margin_of_error, torch.zeros_like(mean))
#     upper_bound = mean + margin_of_error
#
#     lower_bound = lower_bound.reshape(B, N, F)
#     upper_bound = upper_bound.reshape(B, N, F)
#
#     return lower_bound, upper_bound
import torch


def calculate_confidence_intervals(data, confidence_level=0.95):

    B, N, X, F = data.shape
    print(data.shape)
    data = data.permute(0, 1, 3, 2)
    data = data.reshape(B * N * F, X)
    mean = torch.mean(data, axis=1)
    std_dev = torch.std(data, axis=1)

    if confidence_level == 0.95:
        z_score = 1.96  # 95%置信水平的z分数
    elif confidence_level == 0.90:
        z_score = 1.645  # 90%置信水平的z分数
    else:
        z_score = 2.576  # 默认为99%置信水平的z分数

    margin_of_error = z_score * (std_dev / torch.sqrt(torch.tensor(X, dtype=torch.float32)))
    lower_bound = torch.max(mean - margin_of_error, torch.zeros_like(mean))
    upper_bound = mean + margin_of_error

    lower_bound = lower_bound.reshape(B, N, F)
    upper_bound = upper_bound.reshape(B, N, F)

    return lower_bound, upper_bound



def normalization(x, _mean, _std):
    return np.nan_to_num((x - _mean) / _std)


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='SM')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A
    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''
    assert (A - A.transpose()).sum() == 0
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_mean = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_mean - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D), W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0 / np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)
    D = np.diag(1.0 / np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj


def compute_val_loss(net, val_loader, criterion, sw, epoch, mask_matrix, _mean, _std):
    '''
    compute mean loss on validation set
    Parameters
    ----------------------
    net: model
    val_loader: torch.utils.data.utils.DataLoader
    criterion: MSELoss
    sw: tensorboardX.SummaryWriter
    epoch: int, current epoch
    mask_matirx:
    _mean:
    _std:
    Returns
    ----------------------
    val_loss
    '''

    net.train(False)

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []
        type = 'val'

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):
            # labels:(B,N,F,T)
            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1, -2)[:, :, :, :3]  # (B, N, T(1), F)

            labels = labels.transpose(-1, -2)  # (B，N，T(1)，F(3))

            predict_length = labels.shape[2]  # T

            # encode:(B,N,T,d_model)
            encoder_output, encoder_refill, encoder_prob = net.encode(encoder_inputs, _mean, _std)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output, _ = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels, mask_matrix, type) + \
                   criterion(encoder_refill, encoder_inputs[:, :, :, :3], mask_matrix, type)
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

            # delete caches
            del encoder_inputs, decoder_inputs, labels, encoder_output, encoder_refill, decoder_start_inputs, decoder_input_list, predict_output, loss
            torch.cuda.empty_cache()

        print('validation cost time: %.4fs' % (time() - start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _mean, _std, params_path, mask_matrix,
                             mask_or, type):
    # mask_matrix = torch.load(os.path.join('/home/lzm/DTIGNN_UQ_SQ1_qujian_copy/common', 'road_update111111'))
    '''
    Parameters
    --------------------
    net: nn.Module
    data_loader: torch.utils.data.utils.DataLoader
    data_target_tensor: tensor
    epoch: int
    _mean: (1, 1, 11, 1)
    _std: (1, 1, 11, 1)
    params_path: the path for saving the results
    mask_matrix: (N_road, N_road)
    mask_or: dict
    '''
    net.train(False)

    start_time = time()

    lower = []
    upper = []
    mis_var_list = []
    obs_var_list = []
    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        mu_list = []
        a_list = []
        b_list = []
        l_list = []

        input_x = []

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1, -2)[:, :, :, :3]  # (B, N, T, 1)

            labels = labels.transpose(-1, -2)  # (B, N, T, 1)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output, encoder_refill, encoder_prob = net.encode(encoder_inputs, _mean, _std)
            input_x.append(encoder_inputs.cpu().numpy())

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1,
                                   :]  # only get the value of the first timestep，for subsequent data, the predicted values are used for input
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)

                low, up = calculate_confidence_intervals(predict_output[1])
                upper.append(up.cpu())
                lower.append(low.cpu())
                decoder_input_list = [decoder_start_inputs, predict_output]

                out_prob = predict_output[1]
                #这里搞一下上界和下界？

                mu = out_prob[:, :, 0, :].detach().cpu().numpy()
                print(mu)
                print('pred', predict_output[0])
                a = out_prob[:, :, 1, :].detach().cpu().numpy() + 1.0
                b = out_prob[:, :, 2, :].detach().cpu().numpy() + 0.3
                l = out_prob[:, :, 3, :].detach().cpu().numpy() + 1.0

                mu_list.append(mu)
                a_list.append(a)
                b_list.append(b)
                l_list.append(l)


            prediction.append(predict_output[0].detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s, time: %.2fs' % (
                batch_index + 1, loader_length, time() - start_time))

            # delete caches
            del encoder_inputs, decoder_inputs, labels, encoder_output, encoder_refill, decoder_start_inputs, decoder_input_list, predict_output
            torch.cuda.empty_cache()

        mu = np.concatenate(mu_list, axis=0)
        a = np.concatenate(a_list, axis=0)
        b = np.concatenate(b_list, axis=0)
        l = np.concatenate(l_list, axis=0)
        print('mumu', mu.shape)


        lowwww = mu - np.sqrt(b / (a - 1))
        uppppp = mu + np.sqrt(b / (a - 1))

        print('lowwww', lowwww.shape)
        print('uppppp', uppppp.shape)

        data_shape = a.shape
        mask111 = np.ones(data_shape)
        for idx, i in enumerate(mask_matrix):
            if i == 2:
                if len(data_shape) == 3:
                    mask111[:, idx, :] = np.zeros_like(mask111[:, idx, :])
                else:
                    mask111[:, idx, :, :] = np.zeros_like(mask111[:, idx, :, :])
        mask111 = mask111.astype('float32')
        # print(mask111)
        reverse_mask = np.logical_not(mask111).astype('float32')
        # print(reverse_mask)
        # print('mask111', mask111.shape)
        # print('reverse_mask', reverse_mask.shape)

        # obs_idx = list(set(range(0, test_inputs_s.shape[1])) - unknow_set)
        mis_a = a * reverse_mask
        # print('mis_a0', mis_a.shape)
        mis_a = np.mean(mis_a, axis=0)
        # print(mis_a.shape)
        # print(mis_a)
        mis_a_idx = np.where(mis_a >= 1.0)
        # print('mis_a_idx', mis_a_idx)
        mis_a = mis_a[mis_a_idx]
        # print('mis_aaa', mis_a.shape)
        mis_b = b * reverse_mask
        # print('bbbb',mis_b)
        mis_b = np.mean(mis_b, axis=0)[mis_a_idx]
        # print('bbbb',mis_b)
        mis_b = np.mean(mis_b, axis=0)
        # print('bbbb',mis_b)
        mis_l = l * reverse_mask
        mis_l = np.mean(mis_l, axis=0)[mis_a_idx]
        # print('mis_b', mis_b.shape)
        # print('mis_l', mis_l.shape)
        obs_a = a * mask111
        obs_a = np.mean(obs_a, axis=0)
        obs_a_idx = np.where(obs_a >= 1.0)
        obs_a = obs_a[obs_a_idx]
        obs_b = b * mask111
        obs_b = np.mean(obs_b, axis=0)[obs_a_idx]
        obs_l = l * mask111
        obs_l = np.mean(obs_l, axis=0)[obs_a_idx]

        mis_var = mis_b / ((mis_a - 1.0) * mis_l)  # epistemic/ model/prediciton uncertaitnty
        mis_var = mis_var.astype('float32')
        obs_var = obs_b / ((obs_a - 1.0) * obs_l)  # epistemic/ model/prediciton uncertaitnty
        obs_var = obs_var.astype('float32')
        mis_var_list.append(mis_a_idx)
        mis_var_list.append(mis_var)
        obs_var_list.append(obs_a_idx)
        obs_var_list.append(obs_var)
        print('mis_var', mis_var, mis_var.shape)
        obs_var = obs_b / ((obs_a - 1.0) * obs_l)  # epistemic/ model/prediciton uncertaitnty
        obs_var = obs_var.astype('float32')

        print('test time on whole data:%.2fs' % (time() - start_time))
        input_x = np.concatenate(input_x, 0).transpose(0, 1, 3, 2)
        input_x = re_normalization(input_x, _mean, _std)

        prediction = np.concatenate(prediction, 0).transpose(0, 1, 3, 2)  # (batch, N, 1, T')

        print('input:', input_x.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s.pkl' % (epoch, type))
        origin_data = {'input': input_x, 'prediction': prediction, 'data_target_tensor': data_target_tensor,
                       'mask_or': mask_or}
        with open(output_filename, 'wb') as fw:
            pickle.dump(origin_data, fw)

        # calculate error
        excel_list = []

        prediction_length = prediction.shape[3]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = masked_mae_test(data_target_tensor[:, :, :, i], prediction[:, :, :, i], prediction[:, :, :, i].shape,
                                  mask_matrix)
            rmse = masked_rmse_test(data_target_tensor[:, :, :, i], prediction[:, :, :, i],
                                    prediction[:, :, :, i].shape, mask_matrix)
            mape = masked_mape_test(data_target_tensor[:, :, :, i], prediction[:, :, :, i],
                                    prediction[:, :, :, i].shape, mask_matrix)
            print('MAE: %.4f' % (mae))
            print('RMSE: %.4f' % (rmse))
            print('MAPE: %.4f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = masked_mae_test(data_target_tensor, prediction, prediction.shape, mask_matrix)
        rmse = masked_rmse_test(data_target_tensor, prediction, prediction.shape, mask_matrix)
        mape = masked_mape_test(data_target_tensor, prediction, prediction.shape, mask_matrix)
        print("Missing Intersection:")
        for inter in mask_or:
            print(('%s,id:%d') % (mask_or[inter], inter))
        print('all MAE: %.4f' % (mae))
        print('all RMSE: %.4f' % (rmse))
        print('all MAPE: %.4f' % (mape))
        excel_list.extend([mae, rmse, mape])

        # upper = torch.tensor(upper)
        # lower = torch.tensor(lower)

        folder_name = 'res'  # 您可以替换为您想要的文件夹名称
        os.makedirs(folder_name, exist_ok=True)
        torch.save(prediction, os.path.join(folder_name, 'pred'))
        torch.save(upper, os.path.join(folder_name, 'upper'))
        torch.save(lower, os.path.join(folder_name, 'lower'))
        torch.save(data_target_tensor, os.path.join(folder_name, 'data_target_tensor'))
        # print(mis_var_list)
        torch.save(mis_var_list, os.path.join(folder_name, 'mis_var_list'))
        torch.save(obs_var_list, os.path.join(folder_name, 'obs_var_list'))
        torch.save(reverse_mask, os.path.join(folder_name, 'reverse_mask'))
        torch.save(mask111, os.path.join(folder_name, 'mask111'))
        torch.save(mu, os.path.join(folder_name, 'mu'))
        torch.save(uppppp, os.path.join(folder_name, 'uppppp'))
        torch.save(lowwww, os.path.join(folder_name, 'lowwww'))
        # print(excel_list)


def load_graphdata_normY_channel(graph_signal_matrix_filename, DEVICE, batch_size, shuffle=False):
    '''
    load data to generate dataLoader
    Parameters
    ------------------------------------------
    graph_signal_matrix_filename: str
    DEVICE:
    batch_size: int
    Returns
    --------------------------------------------
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)
    '''
    print('load file:', graph_signal_matrix_filename)
    pkl_file = open(graph_signal_matrix_filename, 'rb')
    file_data = pickle.load(pkl_file)

    mask_matrix = file_data['node_update']
    mask_or = file_data['mask_or']

    train_x = file_data['train_x']  # (B, N, F, T)
    train_target = file_data['train_target']  # (B, N, F,1)
    train_timestamp = file_data['train_timestamp']  # (B, 1)

    val_x = file_data['val_x']
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _mean = file_data['mean']  # (1, 1, F, 1)
    _std = file_data['std']  # (1, 1, F, 1)

    # normalize y
    train_target_norm = normalization(train_target, _mean, _std)
    test_target_norm = normalization(test_target, _mean, _std)
    val_target_norm = normalization(val_target, _mean, _std)

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, :, -1:]  # (B, N, F, 1(T))
    # train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :, :-1]),
                                         axis=3)  # (B, N, F, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target[:, :, :3]).type(torch.FloatTensor).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, :, -1:]  # (B, N, 1(F), 1(T))
    # val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :, :-1]), axis=3)  # (B, N, F, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target[:, :, :3]).type(torch.FloatTensor).to(DEVICE)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, :, -1:]  # (B, N, 1(F), 1(T))
    # test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :, :-1]),
                                        axis=3)  # (B, N, F, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target[:, :, :3]).type(torch.FloatTensor).to(DEVICE)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std, mask_matrix, mask_or


if __name__ == '__main__':
    batch_size = 32
    DEVICE = torch.device('cpu')
    graph_signal_matrix_filename = r'../data/hz_4x4/state_data/s30_p1_n4_m2_dataSplit.pkl'
    res = load_graphdata_normY_channel(graph_signal_matrix_filename, DEVICE, batch_size)
    for _, batch_data in enumerate(res[0]):
        encoder_inputs, decoder_inputs, labels = batch_data
        # print(encoder_inputs.shape)
        # print(decoder_inputs.shape)
        # print(labels.shape)
        # print(decoder_inputs)
        # print(labels)
    # print()
    # for _, batch_data in enumerate(res[2]):
    #     encoder_inputs, decoder_inputs, labels = batch_data
    #     print(encoder_inputs.shape)
    #     print(decoder_inputs.shape)
    #     print(labels.shape)

    # print(res[1].shape)