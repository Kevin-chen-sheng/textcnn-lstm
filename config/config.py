import torch


class DefaultConfig(object):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # path
    vector_path = '..\\data\\w2v\\vectors.pkl'
    word2id = "..\\data\\w2v\\word2id.pkl"
    # glove
    vector_path2 = '..\\data\\glove\\vectors.pkl'
    word2id2 = "..\\data\\glove\\word2id.pkl"
    #data
    # train_data_path = "..\\data\\train_data.txt"
    # test_data_path = "..\\data\\test_data.txt"
    train_data_path = "..\\data\\trainnew.txt"
    test_data_path = "..\\data\\testnew.txt"

    train_pos = 250
    train_neg = 250

    test_pos = 82
    test_neg = 2628

    fix_length = 39

    """模型结构参数"""
    model_name = 'Text_cnn'
    freeze = True
    chanel_num = 1
    filter_num = 16
    dropout = 0.3
    num_classes = 2
    static=True
    multichannel=False

    """训练参数"""
    random_seed = 2019
    use_gpu = False
    lr = 0.01
    weight_decay = 0
    num_epochs = 500
    data_shuffle = True
    batch_size = 64
