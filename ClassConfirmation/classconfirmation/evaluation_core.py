import time 
from tqdm.auto import tqdm
import torch
import numpy as np
import os
import json


def create_data_file(total_keep, total_keep_score, total_remove, total_remove_score):
    data = {"keep": [], "remove": []}

    for id_, sc_, in zip(total_keep, total_keep_score):
        data['keep'].append({"id": int(id_), "score_sim": float(sc_)})

    for id_, sc_, in zip(total_remove, total_remove_score):
        data['remove'].append({"id": int(id_), "score_sim": float(sc_)})

    return data


class Evaluator():
    def __init__(self, config, global_rank):
        self.output_dir = config['OUTPUT_DIR']
        self.softmax = torch.nn.Softmax(dim=1)
        self.SUPPORT_AVERAGE_TEST = config['DATASETS']['SUPPORT_AVERAGE_TEST']
        self.global_rank = global_rank

    def eval_model(self, model, dataloaders):
        if self.global_rank == 0:
            print("\n EVALUATING MODEL ")
        since = time.time()
        model.eval()
        total_preds = np.array([], dtype=int)
        total_labels = np.array([], dtype=int)

        for object_crop_instance, positive_support_instance, negative_support_instance in tqdm(dataloaders['test']):
            # If multiples supports -> vote
            if not self.SUPPORT_AVERAGE_TEST:
                raise NotImplementedError
            else:
                querys = object_crop_instance['object_crop'].cuda()
                supports_pos = positive_support_instance['object_crop'].cuda() #(batch, support_n, h,w,c)
                supports_neg = negative_support_instance['object_crop'].cuda()

                labels_gt = (np.concatenate([np.zeros((supports_pos.shape[0])), np.ones((supports_neg.shape[0]))])).astype(int)
                with torch.set_grad_enabled(False):
                    _, _, _, out_pos, out_neg = model(querys, supports_pos, supports_neg)
                    # Softmax 
                    out_pos = self.softmax(out_pos)
                    out_neg = self.softmax(out_neg)

                    _, preds_pos = torch.max(out_pos, 1) 
                    _, preds_neg = torch.max(out_neg, 1) 
                    
                    total_preds = np.concatenate([total_preds, preds_pos.cpu(), preds_neg.cpu()])
                    total_labels = np.concatenate([total_labels, labels_gt])

        time_elapsed = time.time() - since
        acc = np.sum(total_preds == total_labels) / total_labels.shape[0]

        # Precision
        indexes_pr = np.where(total_preds==0)[0]
        pr = np.sum(total_preds[indexes_pr] == total_labels[indexes_pr]) / len(indexes_pr)

        # Recall
        indexes_re = np.where(total_preds==1)[0]
        re = np.sum(total_preds[indexes_re] == total_labels[indexes_re]) / len(indexes_re)
        
        if self.global_rank == 0:
            print('{} Acc: {:.4f}, PR: {:.4f}, RE: {:.4f}'.format('test', acc, pr, re))
            print('Test complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))



    def label_confirmation(self, model, pseudo_loader, save_name):
        print("\n LABEL CONFIRMATION ")
        since = time.time()
        model.eval()
        total_keep = np.array([], dtype=int)
        total_keep_score = np.array([], dtype=float)

        total_remove = np.array([], dtype=int)
        total_remove_score = np.array([], dtype=float)

        for object_crop_instance, positive_support_instance in tqdm(pseudo_loader):
            # If multiples supports -> vote
            if not self.SUPPORT_AVERAGE_TEST:
                raise NotImplementedError
            else:
                querys = object_crop_instance['object_crop'].cuda()
                querys_ids = object_crop_instance['id']
                supports_pos = positive_support_instance['object_crop'].cuda() #(batch, support_n, h,w,c)

                with torch.set_grad_enabled(False):
                    _, _, _, out_pos, _ = model(querys, supports_pos, support_neg=None, precomputed=True)
                    # Softmax 
                    out_pos = self.softmax(out_pos).cpu()

                    _, preds_pos = torch.max(out_pos, 1) 
                    
                    total_keep = np.concatenate([total_keep, querys_ids[preds_pos==0].numpy()])
                    total_keep_score = np.concatenate([total_keep_score, out_pos[preds_pos==0].numpy()[:,0]])

                    total_remove = np.concatenate([total_remove, querys_ids[preds_pos==1].numpy()])
                    total_remove_score = np.concatenate([total_remove_score, out_pos[preds_pos==0].numpy()[:,1]])


        time_elapsed = time.time() - since
        print('Test complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        
        # Save results
        data = create_data_file(total_keep, total_keep_score, total_remove, total_remove_score)
        out_file = os.path.join(self.output_dir, save_name)
        with open(out_file, 'w') as f:
            json.dump(data, f)




        