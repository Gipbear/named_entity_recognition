class TrainingConfig(object):
    """设置lstm训练参数

    --------
    batch_size: 64
    lr: 0.001
    epoches: 30
    print_step: 5

    """
    batch_size = 64
    lr = 0.001
    epoches = 30
    print_step = 5


class LSTMConfig(object):
    """LSTM 配置参数

    --------
    emb_size: 128
    hidden_size: 128
    """
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm 隐向量的维数
