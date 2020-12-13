"""
Main entrypoint file for AM pre-training
"""

import torch
import wandb

from am_v2 import config, dataset, read_data, util, score_main
from am_v2.network import PretrainModel
from am_v2.trainer import Trainer

if __name__ == "__main__":
    # set
    ARGS = config.ARGS

    if not ARGS.debug_mode:
        wandb.init(project=ARGS.project, name=ARGS.name, tags=ARGS.tags, config=ARGS)
    MIN_SIZE = 10
    MAPPING_FILE_PATH = f"am_v2/load/ednet_content_mapping.csv"
    print(MAPPING_FILE_PATH)

    content_mapping = util.read_mapping_item_id(MAPPING_FILE_PATH)
    print("Load score data")
    score_generators = score_main.get_score_generators(content_mapping)

    (
        train_sample_list,
        dev_sample_list,
        test_sample_list,
    ) = read_data.get_user_windows(ARGS.data_file, MIN_SIZE, False)

    num_train_users = len(set([uid for (uid, _) in train_sample_list]))
    num_dev_users = len(set([uid for (uid, _) in dev_sample_list]))
    num_test_users = len(set([uid for (uid, _) in test_sample_list]))

    print(f"# items: {len(content_mapping)}")
    print(f"Train: # users: {num_train_users}, # samples: {len(train_sample_list)}")
    print(f"Val: # users: {num_dev_users}, # samples: {len(dev_sample_list)}")
    print(f"Test: # users: {num_test_users}, # samples: {len(test_sample_list)}")

    train_data = dataset.DataSet(
        f"ednet_train",
        train_sample_list,
        content_mapping,
        ARGS.seq_size,
        is_training=True,
    )
    dev_data = dataset.DataSet(
        f"ednet_dev",
        dev_sample_list,
        content_mapping,
        ARGS.seq_size,
        is_training=False,
    )
    test_data = dataset.DataSet(
        f"ednet_test",
        test_sample_list,
        content_mapping,
        ARGS.seq_size,
        is_training=False,
    )
    train_generator = dataset.make_dataloader(
        dataset=train_data,
        batch_size=ARGS.train_batch,
        shuffle=True,
        num_workers=ARGS.num_workers,
    )
    dev_generator = dataset.make_dataloader(
        dataset=dev_data,
        batch_size=ARGS.test_batch,
        shuffle=False,
        num_workers=ARGS.num_workers,
    )
    test_generator = dataset.make_dataloader(
        dataset=test_data,
        batch_size=ARGS.test_batch,
        shuffle=False,
        num_workers=ARGS.num_workers,
    )
    generators = [train_generator, dev_generator, test_generator]
    for _, batch in enumerate(train_generator):
        print(batch)
        break

    model = PretrainModel(
        q_size=len(content_mapping),
        t_size=3,
        r_size=3,
        p_size=7,
        n_layer=ARGS.num_layer,
        d_model=ARGS.d_model,
        h=ARGS.num_heads,
        dropout=ARGS.dropout,
        device=ARGS.device,
        seq_size=ARGS.seq_size,
        is_feature_based=ARGS.is_feature_based,
        freeze_embedding=ARGS.freeze_embedding,
        freeze_encoder_block=ARGS.freeze_encoder_block,
    ).to(ARGS.device)
    print(model)

    # multi_gpu
    if ARGS.gpu is not None and len(ARGS.gpu) > 1:
        model = torch.nn.DataParallel(model, ARGS.gpu)

    if not ARGS.debug_mode:
        wandb.watch(model)

    # Train_Model
    trainer = Trainer(
        model,
        ARGS.device,
        ARGS.d_model,
        ARGS.debug_mode,
        ARGS.num_epochs,
        ARGS.warmup_steps,
        ARGS.weight_path,
        ARGS.weighted_loss,
        ARGS.lr,
        generators,
        score_generators,
    )
    trainer.train()
