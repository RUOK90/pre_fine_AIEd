import dataset_util, score_dataset, ednet_dataset
from am_network import PretrainModel, ScoreModel
from electra import ElectraAIEdPretrainModel, ElectraAIEdFinetuneModel
from trainer import Trainer
from finetune_trainer import FineTuneTrainer
from config import *


if __name__ == "__main__":
    # get dataset
    q_info_dic = dataset_util.get_q_info_dic(ARGS.question_info_path)
    pretrain_dataloaders = finetune_dataloaders = None
    if ARGS.train_mode == "both":
        pretrain_dataloaders = ednet_dataset.get_dataloaders(
            q_info_dic, ARGS.pretrain_base_path
        )
        if ARGS.downstream_task == "score":
            finetune_dataloaders = score_dataset.get_dataloaders(
                q_info_dic, ARGS.score_base_path
            )
    elif ARGS.train_mode == "pretrain_only":
        pretrain_dataloaders = ednet_dataset.get_dataloaders(
            q_info_dic, ARGS.pretrain_base_path
        )
    elif ARGS.train_mode == "finetune_only":
        if ARGS.downstream_task == "score":
            finetune_dataloaders = score_dataset.get_dataloaders(
                q_info_dic, ARGS.score_base_path
            )

    # get model
    if ARGS.model == "am":
        pretrain_model = PretrainModel()
        if ARGS.downstream_task == "score":
            finetune_model = ScoreModel()
    elif ARGS.model == "electra":
        pretrain_model = ElectraAIEdPretrainModel()
        if ARGS.downstream_task == "score":
            finetune_model = ElectraAIEdFinetuneModel()

    finetune_trainer = FineTuneTrainer(finetune_model, finetune_dataloaders)
    trainer = Trainer(pretrain_model, pretrain_dataloaders, finetune_trainer)
    trainer._train()
