import time 
from tqdm.auto import tqdm
import torch
from importlib import import_module
import os

class Trainer():
    """
    Trainer class for any model

    TO DO:
     - 

    Parameters
    ----------
    config : dict
        Loaded yaml
    
    Returns
    -------

    """

    def __init__(self, config, evaluator, global_rank):
        self.config = config
        self.epochs = config['SOLVER']['EPOCH'] 
        self.base_lr = config['SOLVER']['BASE_LR']
        self.iter_history_period = config['SOLVER']['ITER_HISTORY_PERIOD']
        self.output_dir = config['OUTPUT_DIR']
        self.freeze_backbone = config['MODEL']['FREEZE_BACKBONE']
        
        self.optimizer = getattr(import_module(config['SOLVER']['OPTIMIZER']['TYPE'].rsplit(".",maxsplit=1)[0]), config['SOLVER']['OPTIMIZER']['TYPE'].rsplit(".",maxsplit=1)[1])
        self.weight_decay = config['SOLVER']['OPTIMIZER']['WEIGHT_DECAY']
        if config['SOLVER']['OPTIMIZER']['TYPE'] == "torch.optim.SGD":
            self.momentum = config['SOLVER']['OPTIMIZER']['MOMENTUM']
        
        self.scheduler = getattr(import_module(config['SOLVER']['SCHEDULER']['TYPE'].rsplit(".",maxsplit=1)[0]), config['SOLVER']['SCHEDULER']['TYPE'].rsplit(".",maxsplit=1)[1])
        self.step_size = config['SOLVER']['SCHEDULER']['STEP_SIZE']
        self.gamma = config['SOLVER']['SCHEDULER']['GAMMA']


        self.loss1_active = config['SOLVER']['LOSS1']['ACTIVE']
        if self.loss1_active:
            self.loss1 = getattr(import_module(config['SOLVER']['LOSS1']['TYPE'].rsplit(".",maxsplit=1)[0]), config['SOLVER']['LOSS1']['TYPE'].rsplit(".",maxsplit=1)[1])
            self.loss1_reduction = config['SOLVER']['LOSS1']['REDUCTION']
            self.loss1_weight = config['SOLVER']['LOSS1']['WEIGHT']
            self.loss1_margin = config['SOLVER']['LOSS1']['MARGIN']

        self.loss2 = getattr(import_module(config['SOLVER']['LOSS2']['TYPE'].rsplit(".",maxsplit=1)[0]), config['SOLVER']['LOSS2']['TYPE'].rsplit(".",maxsplit=1)[1])
        self.loss2_reduction = config['SOLVER']['LOSS2']['REDUCTION']
        self.loss2_weight = config['SOLVER']['LOSS2']['WEIGHT']

        self.evaluator = evaluator
        self.global_rank = global_rank


    def train(self, model, dataloaders):
        """
        TO DO:
        - 
        """
        # Freeze model if so
        if self.freeze_backbone:
            for p in model.module.backbone.parameters():
                p.requires_grad = False

        # Build optimizer
        if self.optimizer is torch.optim.SGD:
            optimizer = self.optimizer(model.parameters(), lr=self.base_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer(model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)

        scheduler = self.scheduler(optimizer, self.step_size, self.gamma) 

        # Build losses
        if self.loss1_active:
            loss_1 = self.loss1(reduction=self.loss1_reduction, margin=self.loss1_margin)  # Triplet (only fot the backbone)
        loss_2 = self.loss2(reduction=self.loss2_reduction)  # CrossEntropy final clasification

        model.train()

        for epoch in range(self.epochs):
            if self.global_rank == 0:
                print('\nEpoch {}/{}'.format(epoch+1, self.epochs))
                print('-' * 15)
            iterator = tqdm(dataloaders['train'])
            it_n = 0
            for object_crop_instance, positive_support_instance, negative_support_instance in iterator:
                optimizer.zero_grad()
                querys = object_crop_instance['object_crop'].cuda()
                supports_pos = positive_support_instance['object_crop'].cuda()  #(batch, support_n, h,w,c)
                supports_neg = negative_support_instance['object_crop'].cuda()

                with torch.set_grad_enabled(True):
                    query_emb, support_pos_emb, support_neg_emb, out_pos, out_neg = model(querys, supports_pos, supports_neg)

                    losses = {}
                    if self.loss1_active:
                        loss_backbone = loss_1(query_emb, support_pos_emb, support_neg_emb)
                        losses['backbone_loss'] = loss_backbone * self.loss1_weight
                    
                    # To only 1 single unit at the end (sigmoiud + BCE cross entropy)
                    #loss_head = loss_2(torch.cat((out_pos,out_neg),0), torch.cat((torch.ones((out_pos.shape[0],1)), torch.zeros((out_neg.shape[0],1))),0).to(self.DEVICE))
                    
                    # TO 2 units at end (cross entropy -> class0 same, class1 diferent) 
                    gt0 = torch.zeros((out_pos.shape[0],2))
                    gt0[:,0] = 1.0
                    gt1 = torch.zeros((out_neg.shape[0],2))
                    gt1[:,1] = 1.0
                    gt_ = torch.cat((gt0,gt1),0).cuda()

                    loss_head = loss_2(torch.cat((out_pos,out_neg),0), gt_)
                    losses['header_loss'] = loss_head * self.loss2_weight

                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()

                if it_n % self.iter_history_period == 0:
                    message = f'total_loss: {float(total_loss.detach().cpu()):.4f}'
                    for l,v in losses.items():
                        message += f'   {l}: {v.detach().cpu():.4f}'
                    message += f"   learning_rate: {scheduler.get_last_lr()}"
                    if self.global_rank == 0:
                        print(message)
                it_n += 1
            scheduler.step()
            # After each epoch -> save model + evaluation
            save_dir_ = os.path.join(self.output_dir, f"final_model_epoch_{epoch}.pth")
            print(f"Save model in {save_dir_}")
            torch.save(model.state_dict(), save_dir_)
            # if epoch == self.epochs-1:
            #     self.evaluator.eval_model(model, dataloaders)
            self.evaluator.eval_model(model, dataloaders)



