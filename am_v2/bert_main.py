import torch
import wandb
import am_v2.config as config
from am_v2.bert_dataset import BERTDataSet, load_data
from am_v2.bert_trainer import BERTTrainer
from am_v2.network import BERT

if __name__ == '__main__':
    ARGS = config.ARGS

    if not ARGS.debug_mode:
        wandb.init(project=ARGS.project, name=ARGS.name, tags=ARGS.tags, config=ARGS)
    qid_to_tokens, sample_list = load_data()
    num_vocab = max([max(token_list) for qid, token_list in qid_to_tokens.items()])
    train_data = BERTDataSet(
        num_vocab,
        qid_to_tokens,
        sample_list,
        sequence_size=100,
        is_training=True,
    )
    train_generator = torch.utils.data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=ARGS.train_batch,
        num_workers=ARGS.num_workers,
    )

    model = BERT(
        num_vocab+1,
        n_layer=2,
        d_model=256,
        h=8,
        dropout=0.1,
        device=ARGS.device,
        seq_size=100,
        is_feature_based=False,
        freeze_embedding=False,
        freeze_encoder_block=False,
    ).to(ARGS.device)
    trainer = BERTTrainer(
        model,
        ARGS.device,
        ARGS.d_model,
        ARGS.debug_mode,
        ARGS.num_epochs,
        ARGS.lr,
        ARGS.warmup_steps,
        [train_generator, None, None],
        qid_to_tokens,
    )
    trainer.train()
