import dataset_util, score_dataset, ednet_dataset
from am_network import PretrainModel, ScoreModel
from trainer import Trainer
from finetune_trainer import FineTuneTrainer
from config import *


if __name__ == "__main__":
    q_info_dic = dataset_util.get_q_info_dic(ARGS.question_info_path)
    pretrain_dataloaders = ednet_dataset.get_dataloaders(
        q_info_dic, ARGS.pretrain_base_path
    )

    if ARGS.model == "am":
        pretrain_model = PretrainModel(num_qs=len(q_info_dic))
        if ARGS.downstream_task == "score":
            finetune_model = ScoreModel(num_qs=len(q_info_dic))
            finetune_dataloaders = score_dataset.get_dataloaders(
                q_info_dic, ARGS.score_base_path
            )

    elif ARGS.model == "electra":
        ""

    finetune_trainer = FineTuneTrainer(finetune_model, finetune_dataloaders)
    trainer = Trainer(pretrain_model, pretrain_dataloaders, finetune_trainer)
    trainer._train()
