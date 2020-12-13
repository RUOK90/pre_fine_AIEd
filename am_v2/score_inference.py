import sys
from score_trainer import Trainer
# from network import RUNs
from score_network import Model
import score_dataset
from torch.utils import data
import torch.nn as nn
import os
print(os.getcwd())
sys.path.append("..")
from config import args, print_args
import util
import utils
import csv
import torch


def load_weight(weight_path, model, device):
    weight = torch.load(weight_path, map_location=device)

    for name, parm in model.named_parameters():
        if name in weight.keys():
            parm.data.copy_(weight[name])


if __name__ == '__main__':
    # set
    mapping_file_path = 'load/content_dict.csv'
    max_elapsed_time = 300

    base_path = '/shared/AAAI20/score_data/14d_10q/'
    user_data_path = base_path + 'response/'
    # user_data_path = base_path + 'tmp_response/'

    print_args(args, is_write=False)
    print(user_data_path, mapping_file_path)
    old_to_new_item_id = util.read_mapping_item_id(mapping_file_path)
    dict_start_time_index = util.get_dict_start_time_index()

    train_user_id_list = utils.load_csv(base_path + 'train_user_list.csv')
    val_user_id_list = utils.load_csv(base_path + 'validation_user_list.csv')
    test_user_id_list = utils.load_csv(base_path + 'test_user_list.csv')
    # test_user_id_list = utils.load_csv(base_path + 'tmp_user_list.csv')
    print(f'# of train_users: {len(train_user_id_list)}, # of val_users: {len(val_user_id_list)}, # of test_users: {len(test_user_id_list)}')
    print(args)
    # val_data = score_dataset.DataSet(False, False, 1, user_data_path, old_to_new_item_id, val_user_id_list,
    #                                  args.seq_size)
    test_data = score_dataset.DataSet(False, False, 1, user_data_path, old_to_new_item_id, test_user_id_list, args.seq_size)

    print(test_data)

    train_generator = None
    val_generator = None
    # val_generator = data.DataLoader(dataset=val_data, shuffle=False, batch_size=args.test_batch,
    #                                 num_workers=args.num_workers)
    test_generator = data.DataLoader(dataset=test_data, shuffle=False, batch_size=args.test_batch,
                                    num_workers=args.num_workers)

    model = Model(q_size=len(old_to_new_item_id) + 1, start_time_size=len(dict_start_time_index) + 1,
                  elapsed_time_size=max_elapsed_time + 2,
                  r_size=4, p_size=9, n_layer=args.num_layer, d_model=args.d_model,
                  h=args.num_heads, dropout=args.dropout, device=args.device,
                  max_len=512, is_feature_based=args.is_feature_based).to(args.device)

    trainer = Trainer(model, args.debug_mode, args.num_epochs, args.weight_path, args.d_model, args.warmup_steps,
                      args.lr, args.device, train_generator, val_generator, test_generator, args.lambda1, args.lambda2)

    TEST_MODE = 'test'
    load_weight('pre_train_weight/am11_tp24_49.pt', model, args.device)
    # trainer._inference(test_generator, mode=TEST_MODE)

    with torch.no_grad():
        model.eval()
        trainer._inference(test_generator, mode=TEST_MODE)



