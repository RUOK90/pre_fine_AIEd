from datasets import pretrain_dataset, score_dataset, dataset_util
from models.am_network import PretrainModel, ScoreModel
from models.electra import ElectraAIEdPretrainModel, ElectraAIEdFinetuneModel
from trainer import Trainer
from finetune_trainer import FineTuneTrainer
from config import *


if __name__ == "__main__":
    # get dataset
    q_info_dic = dataset_util.get_q_info_dic(ARGS.question_info_path)
    user_inters_dic = dataset_util.get_user_interactions_dic(ARGS.interaction_base_path)
    pretrain_dataloaders = finetune_dataloaders = None
    if ARGS.train_mode == "both":
        pretrain_dataloaders = pretrain_dataset.get_dataloaders(
            q_info_dic, user_inters_dic, ARGS.pretrain_base_path
        )
        if ARGS.downstream_task == "score":
            finetune_dataloaders = score_dataset.get_dataloaders(
                q_info_dic, user_inters_dic, ARGS.score_base_path
            )
    elif ARGS.train_mode == "pretrain_only":
        pretrain_dataloaders = pretrain_dataset.get_dataloaders(
            q_info_dic, user_inters_dic, ARGS.pretrain_base_path
        )
    elif (
        ARGS.train_mode == "finetune_only"
        or ARGS.train_mode == "finetune_only_from_pretrained_weight"
    ):
        if ARGS.downstream_task == "score":
            finetune_dataloaders = score_dataset.get_dataloaders(
                q_info_dic, user_inters_dic, ARGS.score_base_path
            )

    # get model
    if ARGS.model == "am":
        pretrain_model = PretrainModel()
        finetune_model = ScoreModel()
    elif (
        ARGS.model == "electra"
        or ARGS.model == "electra-reformer"
        or ARGS.model == "electra-performer"
    ):
        pretrain_model = ElectraAIEdPretrainModel()
        finetune_model = ElectraAIEdFinetuneModel()

    finetune_trainer = FineTuneTrainer(finetune_model, finetune_dataloaders)
    trainer = Trainer(pretrain_model, pretrain_dataloaders, finetune_trainer)
    trainer._train()
