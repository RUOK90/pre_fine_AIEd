import sys
import wandb
from trainer import Trainer
from network import RUNs
import read_data
import dataset
from torch.utils import data
from tqdm import tqdm

sys.path.append("..")
from config import args, print_args
import util
import torch


def set_weight(model, weight_path):
    pre_trained_weight = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(pre_trained_weight)

    return model



if __name__ == '__main__':
    mapping_file_path = 'load/mapping_part5.csv'
    if not args.debug_mode:
        split_user_path = 'load/'
        user_data_path = '/root/workspace/cikm_data_part5/'
        write_path = ''
    else:
        split_user_path = 'load/tmp_'
        user_data_path = '../user_data/unlabeled/'
        write_path = '../user_data/tmp/'
        weight_path = 'weight/test/0.pt'

    print_args(args, is_write=False)
    print(user_data_path, mapping_file_path)
    old_to_new_item_id = util.read_mapping_item_id(mapping_file_path)

    test_user_id2csv_path, test_sample_list = read_data.get_sample_list_from_csv(f'{split_user_path}test_user.csv', 1, False, 50, 200, 21)
    test_data = dataset.DataSet('val', user_data_path, test_user_id2csv_path, test_sample_list, old_to_new_item_id, args.seq_size)
    test_generator = data.DataLoader(dataset=test_data, shuffle=False, batch_size=args.test_batch, num_workers=args.num_workers)

    print(f'# of items: {len(old_to_new_item_id)}')
    print(f'Test: # of users: {len(test_user_id2csv_path)}, # of samples: {len(test_sample_list)}')


    # model
    model = RUNs(len(old_to_new_item_id), args.item_feature_dim, args.cf_mode, args.device).to(args.device)

    # weight load
    model = set_weight(model, weight_path)

    # inference
    correct_count = 0
    total_count = 0

    predicted_label_list = []
    true_label_list = []
    disc_list = []
    user_list = []
    index_list = []

    for x, y, user_id, item_index in tqdm(test_generator):
        predicted_label, true_label, vae_loss, bce_loss = model(x, y)

        true_label_list += list(true_label.data.cpu().numpy())
        predicted_label_list += list(predicted_label.data.cpu().numpy())
        output2binary = (predicted_label > 0.5).float()
        disc_list += list(output2binary.data.cpu().numpy())
        user_list += list(user_id)
        index_list += list(item_index.data.cpu().numpy())

        correct_count += float((output2binary == true_label).float().sum())
        total_count += len(output2binary)

    print(f'End inference, ACC: {correct_count/total_count}')

    user_dict = {}
    for i in range(len(index_list)):
        index = index_list[i]
        user_id = user_list[i]

        if user_id not in user_dict:
            user_dict[user_id] = {}

        label = true_label_list[i]
        prob = predicted_label_list[i]
        disc = disc_list[i]

        user_dict[user_id][index] = [disc, prob, label]

    for user in user_dict:
        target_path = f'{user_data_path}{user}'
        cur_write_path = f'{write_path}{user}'
        line_list = util.load_csv(target_path)

        write_line = []
        with open(cur_write_path, 'w') as f:

            for index in sorted(user_dict[user]):
                row = line_list[index].split(',')
                time_date = row[0]
                item_id = row[1]
                disc = int(user_dict[user][index][0])
                prob = user_dict[user][index][1]
                label = int(user_dict[user][index][2])

                write_row = f'{time_date},{item_id},{disc},{prob},{label}'
                f.write(write_row + '\n')


