import torch
from torch.nn import functional as F



class LabelConfirmation(torch.nn.Module):
    """
    Label Confirmation model (base) (see paper figure)


    Parameters
    ----------
    backbone : Pytorch model
        MAE or DINO
    backbone_dimension: int
        Size of the final vector (features). In DINO cls token, in MAE àverage pooling
    token_average_pooling: Bool
        Perform average pooling of the last hidden space
    """

    def __init__(self, backbone, backbone_dimension, token_average_pooling=False):
        super(LabelConfirmation, self).__init__()

        self.token_average_pooling = token_average_pooling
        self.backbone = backbone
        self.cls1 = torch.nn.Linear(backbone_dimension*2, 1024)
        self.cls2 = torch.nn.Linear(1024, 1024)
        self.cls3 = torch.nn.Linear(1024, 2)
        #self.sigmoid = torch.nn.Sigmoid()  # If softmax this is apply by CrossEntropyLoss calculation

    def forward(self, query, support_pos, support_neg=None, precomputed=False):
        # Query features. Always 1
        query_out = self.backbone(query).last_hidden_state
        support_pos_out = support_pos  # Inicialization in the case 
        support_neg_out = None

        if not precomputed:
            # Positive suport features. If support size > 1, mean over them.
            support_pos_aux = torch.reshape(support_pos, (support_pos.shape[0]* support_pos.shape[1], support_pos.shape[2],support_pos.shape[3],support_pos.shape[4]))
            support_pos_out = self.backbone(support_pos_aux).last_hidden_state
            support_pos_out = torch.reshape(support_pos_out, (support_pos.shape[0], support_pos.shape[1], support_pos_out.shape[1], support_pos_out.shape[2]))
            support_pos_out = torch.mean(support_pos_out, dim=1)

            if support_neg is not None:
                support_neg_aux = torch.reshape(support_neg, (support_neg.shape[0]* support_neg.shape[1], support_neg.shape[2],support_neg.shape[3],support_neg.shape[4]))
                support_neg_out = self.backbone(support_neg_aux).last_hidden_state
                support_neg_out = torch.reshape(support_neg_out, (support_neg.shape[0], support_neg.shape[1], support_neg_out.shape[1], support_neg_out.shape[2]))
                support_neg_out = torch.mean(support_neg_out, dim=1)

        
        if self.token_average_pooling:
            query_out = torch.mean(query_out[:,1:,:], dim=1)
            if not precomputed:
                support_pos_out = torch.mean(support_pos_out, dim=1)
                if support_neg is not None:
                    support_neg_out = torch.mean(support_neg_out, dim=1)
        else:
            query_out = query_out[:,0,:]
            if not precomputed:
                support_pos_out = support_pos_out[:,0,:]
                if support_neg is not None:
                    support_neg_out = support_neg_out[:,0,:]
            
        # Positive branch
        concat = torch.cat((query_out, support_pos_out), 1)
        out_pos = F.relu(self.cls1(concat))
        out_pos = F.relu(self.cls2(out_pos))
        out_pos = self.cls3(out_pos)
        #out_pos = self.sigmoid(out_pos)

        # Negative branch
        out_neg = None
        if support_neg is not None:
            concat = torch.cat((query_out, support_neg_out), 1)
            out_neg = F.relu(self.cls1(concat))
            out_neg = F.relu(self.cls2(out_neg))
            out_neg = self.cls3(out_neg)
            #out_neg = self.sigmoid(out_neg)

        return query_out, support_pos_out, support_neg_out, out_pos, out_neg
    


class LabelConfirmation2(torch.nn.Module):
    """
    Label Confirmation model. Con capas fully connected despues de obtener backbone embedding y antes de concatenar


    Parameters
    ----------
    backbone : Pytorch model
        MAE or DINO
    backbone_dimension: int
        Size of the final vector (features). In DINO cls token, in MAE àverage pooling
    token_average_pooling: Bool
        Perform average pooling of the last hidden space
    """

    def __init__(self, backbone, backbone_dimension, token_average_pooling=False):
        super(LabelConfirmation2, self).__init__()

        self.token_average_pooling = token_average_pooling
        self.backbone = backbone
        
        # Before concatenation
        self.emb1 = torch.nn.Linear(backbone_dimension, backbone_dimension)

        # After concatenation
        self.cls1 = torch.nn.Linear(backbone_dimension*2, 1024)
        self.cls2 = torch.nn.Linear(1024, 1024)
        self.cls3 = torch.nn.Linear(1024, 2)
        #self.sigmoid = torch.nn.Sigmoid()  # If softmax this is apply by CrossEntropyLoss calculation

    def forward(self, query, support_pos, support_neg=None, precomputed=False):
        # Query features. Always 1
        query_out = self.backbone(query).last_hidden_state
        support_pos_out = support_pos  # Inicialization in the case 
        support_neg_out = None

        if not precomputed:
            # Positive suport features. If support size > 1, mean over them.
            support_pos_aux = torch.reshape(support_pos, (support_pos.shape[0]* support_pos.shape[1], support_pos.shape[2],support_pos.shape[3],support_pos.shape[4]))
            support_pos_out = self.backbone(support_pos_aux).last_hidden_state
            support_pos_out = torch.reshape(support_pos_out, (support_pos.shape[0], support_pos.shape[1], support_pos_out.shape[1], support_pos_out.shape[2]))
            support_pos_out = torch.mean(support_pos_out, dim=1)


            if support_neg is not None:
                support_neg_aux = torch.reshape(support_neg, (support_neg.shape[0]* support_neg.shape[1], support_neg.shape[2],support_neg.shape[3],support_neg.shape[4]))
                support_neg_out = self.backbone(support_neg_aux).last_hidden_state
                support_neg_out = torch.reshape(support_neg_out, (support_neg.shape[0], support_neg.shape[1], support_neg_out.shape[1], support_neg_out.shape[2]))
                support_neg_out = torch.mean(support_neg_out, dim=1)

        
        if self.token_average_pooling:
            query_out = torch.mean(query_out[:,1:,:], dim=1)
            if not precomputed:
                support_pos_out = torch.mean(support_pos_out, dim=1)
                if support_neg is not None:
                    support_neg_out = torch.mean(support_neg_out, dim=1)
        else:
            query_out = query_out[:,0,:]
            if not precomputed:
                support_pos_out = support_pos_out[:,0,:]
                if support_neg is not None:
                    support_neg_out = support_neg_out[:,0,:]
        
        
        # Embeddings fully connected
        query_out = self.emb1(query_out)

        # Positive branch
        support_pos_out = self.emb1(support_pos_out)
        concat = torch.cat((query_out, support_pos_out), 1)
        out_pos = F.relu(self.cls1(concat))
        out_pos = F.relu(self.cls2(out_pos))
        out_pos = self.cls3(out_pos)
        #out_pos = self.sigmoid(out_pos)

        # Negative branch
        out_neg = None
        if support_neg is not None:
            support_neg_out = self.emb1(support_neg_out)
            concat = torch.cat((query_out, support_neg_out), 1)
            out_neg = F.relu(self.cls1(concat))
            out_neg = F.relu(self.cls2(out_neg))
            out_neg = self.cls3(out_neg)
            #out_neg = self.sigmoid(out_neg)

        return query_out, support_pos_out, support_neg_out, out_pos, out_neg



class LabelConfirmation3(torch.nn.Module):
    """
    Label Confirmation model. 
    Con capas fully connected despues de obtener backbone embedding y antes de concatenar
    Con sola 2 capas en cabecera


    Parameters
    ----------
    backbone : Pytorch model
        MAE or DINO
    backbone_dimension: int
        Size of the final vector (features). In DINO cls token, in MAE àverage pooling
    token_average_pooling: Bool
        Perform average pooling of the last hidden space
    """

    def __init__(self, backbone, backbone_dimension, token_average_pooling=False):
        super(LabelConfirmation3, self).__init__()

        self.token_average_pooling = token_average_pooling
        self.backbone = backbone
        
        # Before concatenation
        self.emb1 = torch.nn.Linear(backbone_dimension, backbone_dimension)

        # After concatenation
        self.cls1 = torch.nn.Linear(backbone_dimension*2, 1024)
        self.cls2 = torch.nn.Linear(1024, 2)
        #self.sigmoid = torch.nn.Sigmoid()  # If softmax this is apply by CrossEntropyLoss calculation

    def forward(self, query, support_pos, support_neg=None, precomputed=False):
        # Query features. Always 1
        query_out = self.backbone(query).last_hidden_state
        support_pos_out = support_pos  # Inicialization in the case 
        support_neg_out = None

        if not precomputed:
            # Positive suport features. If support size > 1, mean over them.
            support_pos_aux = torch.reshape(support_pos, (support_pos.shape[0]* support_pos.shape[1], support_pos.shape[2],support_pos.shape[3],support_pos.shape[4]))
            support_pos_out = self.backbone(support_pos_aux).last_hidden_state
            support_pos_out = torch.reshape(support_pos_out, (support_pos.shape[0], support_pos.shape[1], support_pos_out.shape[1], support_pos_out.shape[2]))
            support_pos_out = torch.mean(support_pos_out, dim=1)


            if support_neg is not None:
                support_neg_aux = torch.reshape(support_neg, (support_neg.shape[0]* support_neg.shape[1], support_neg.shape[2],support_neg.shape[3],support_neg.shape[4]))
                support_neg_out = self.backbone(support_neg_aux).last_hidden_state
                support_neg_out = torch.reshape(support_neg_out, (support_neg.shape[0], support_neg.shape[1], support_neg_out.shape[1], support_neg_out.shape[2]))
                support_neg_out = torch.mean(support_neg_out, dim=1)

        
        if self.token_average_pooling:
            query_out = torch.mean(query_out[:,1:,:], dim=1)
            if not precomputed:
                support_pos_out = torch.mean(support_pos_out, dim=1)
                if support_neg is not None:
                    support_neg_out = torch.mean(support_neg_out, dim=1)
        else:
            query_out = query_out[:,0,:]
            if not precomputed:
                support_pos_out = support_pos_out[:,0,:]
                if support_neg is not None:
                    support_neg_out = support_neg_out[:,0,:]
        
        
        # Embeddings fully connected
        query_out = self.emb1(query_out)

        # Positive branch
        support_pos_out = self.emb1(support_pos_out)
        concat = torch.cat((query_out, support_pos_out), 1)
        out_pos = F.relu(self.cls1(concat))
        out_pos = self.cls2(out_pos)
        #out_pos = self.sigmoid(out_pos)

        # Negative branch
        out_neg = None
        if support_neg is not None:
            support_neg_out = self.emb1(support_neg_out)
            concat = torch.cat((query_out, support_neg_out), 1)
            out_neg = F.relu(self.cls1(concat))
            out_neg = self.cls2(out_neg)
            #out_neg = self.sigmoid(out_neg)

        return query_out, support_pos_out, support_neg_out, out_pos, out_neg




class LabelConfirmation4(torch.nn.Module):
    """
    Label Confirmation model. 
    Con capas fully connected despues de obtener backbone embedding y antes de concatenar
    Con sola 1 capas en cabecera


    Parameters
    ----------
    backbone : Pytorch model
        MAE or DINO
    backbone_dimension: int
        Size of the final vector (features). In DINO cls token, in MAE àverage pooling
    token_average_pooling: Bool
        Perform average pooling of the last hidden space
    """

    def __init__(self, backbone, backbone_dimension, token_average_pooling=False):
        super(LabelConfirmation4, self).__init__()

        self.token_average_pooling = token_average_pooling
        self.backbone = backbone
        
        # Before concatenation
        self.emb1 = torch.nn.Linear(backbone_dimension, backbone_dimension)

        # After concatenation
        self.cls1 = torch.nn.Linear(backbone_dimension*2, 2)
        #self.sigmoid = torch.nn.Sigmoid()  # If softmax this is apply by CrossEntropyLoss calculation

    def forward(self, query, support_pos, support_neg=None, precomputed=False):
        # Query features. Always 1
        query_out = self.backbone(query).last_hidden_state
        support_pos_out = support_pos  # Inicialization in the case 
        support_neg_out = None

        if not precomputed:
            # Positive suport features. If support size > 1, mean over them.
            support_pos_aux = torch.reshape(support_pos, (support_pos.shape[0]* support_pos.shape[1], support_pos.shape[2],support_pos.shape[3],support_pos.shape[4]))
            support_pos_out = self.backbone(support_pos_aux).last_hidden_state
            support_pos_out = torch.reshape(support_pos_out, (support_pos.shape[0], support_pos.shape[1], support_pos_out.shape[1], support_pos_out.shape[2]))
            support_pos_out = torch.mean(support_pos_out, dim=1)


            if support_neg is not None:
                support_neg_aux = torch.reshape(support_neg, (support_neg.shape[0]* support_neg.shape[1], support_neg.shape[2],support_neg.shape[3],support_neg.shape[4]))
                support_neg_out = self.backbone(support_neg_aux).last_hidden_state
                support_neg_out = torch.reshape(support_neg_out, (support_neg.shape[0], support_neg.shape[1], support_neg_out.shape[1], support_neg_out.shape[2]))
                support_neg_out = torch.mean(support_neg_out, dim=1)

        
        if self.token_average_pooling:
            query_out = torch.mean(query_out[:,1:,:], dim=1)
            if not precomputed:
                support_pos_out = torch.mean(support_pos_out, dim=1)
                if support_neg is not None:
                    support_neg_out = torch.mean(support_neg_out, dim=1)
        else:
            query_out = query_out[:,0,:]
            if not precomputed:
                support_pos_out = support_pos_out[:,0,:]
                if support_neg is not None:
                    support_neg_out = support_neg_out[:,0,:]
        
        
        # Embeddings fully connected
        query_out = self.emb1(query_out)

        # Positive branch
        support_pos_out = self.emb1(support_pos_out)
        concat = torch.cat((query_out, support_pos_out), 1)
        out_pos = self.cls1(concat)
        #out_pos = self.sigmoid(out_pos)

        # Negative branch
        out_neg = None
        if support_neg is not None:
            support_neg_out = self.emb1(support_neg_out)
            concat = torch.cat((query_out, support_neg_out), 1)
            out_neg = self.cls1(concat)
            #out_neg = self.sigmoid(out_neg)

        return query_out, support_pos_out, support_neg_out, out_pos, out_neg
    

class LabelConfirmation5(torch.nn.Module):
    """
    Label Confirmation model. 
    Con sola 2 capas en cabecera


    Parameters
    ----------
    backbone : Pytorch model
        MAE or DINO
    backbone_dimension: int
        Size of the final vector (features). In DINO cls token, in MAE àverage pooling
    token_average_pooling: Bool
        Perform average pooling of the last hidden space
    """

    def __init__(self, backbone, backbone_dimension, token_average_pooling=False):
        super(LabelConfirmation5, self).__init__()

        self.token_average_pooling = token_average_pooling
        self.backbone = backbone
        
        # After concatenation
        self.cls1 = torch.nn.Linear(backbone_dimension*2, 1024)
        self.cls2 = torch.nn.Linear(1024, 2)
        #self.sigmoid = torch.nn.Sigmoid()  # If softmax this is apply by CrossEntropyLoss calculation

    def forward(self, query, support_pos, support_neg=None, precomputed=False):
        # Query features. Always 1
        query_out = self.backbone(query).last_hidden_state
        support_pos_out = support_pos  # Inicialization in the case 
        support_neg_out = None

        if not precomputed:
            # Positive suport features. If support size > 1, mean over them.
            support_pos_aux = torch.reshape(support_pos, (support_pos.shape[0]* support_pos.shape[1], support_pos.shape[2],support_pos.shape[3],support_pos.shape[4]))
            support_pos_out = self.backbone(support_pos_aux).last_hidden_state
            support_pos_out = torch.reshape(support_pos_out, (support_pos.shape[0], support_pos.shape[1], support_pos_out.shape[1], support_pos_out.shape[2]))
            support_pos_out = torch.mean(support_pos_out, dim=1)


            if support_neg is not None:
                support_neg_aux = torch.reshape(support_neg, (support_neg.shape[0]* support_neg.shape[1], support_neg.shape[2],support_neg.shape[3],support_neg.shape[4]))
                support_neg_out = self.backbone(support_neg_aux).last_hidden_state
                support_neg_out = torch.reshape(support_neg_out, (support_neg.shape[0], support_neg.shape[1], support_neg_out.shape[1], support_neg_out.shape[2]))
                support_neg_out = torch.mean(support_neg_out, dim=1)

        
        if self.token_average_pooling:
            query_out = torch.mean(query_out[:,1:,:], dim=1)
            if not precomputed:
                support_pos_out = torch.mean(support_pos_out, dim=1)
                if support_neg is not None:
                    support_neg_out = torch.mean(support_neg_out, dim=1)
        else:
            query_out = query_out[:,0,:]
            if not precomputed:
                support_pos_out = support_pos_out[:,0,:]
                if support_neg is not None:
                    support_neg_out = support_neg_out[:,0,:]
        
        
        # Positive branch
        concat = torch.cat((query_out, support_pos_out), 1)
        out_pos = F.relu(self.cls1(concat))
        out_pos = self.cls2(out_pos)
        #out_pos = self.sigmoid(out_pos)

        # Negative branch
        out_neg = None
        if support_neg is not None:
            concat = torch.cat((query_out, support_neg_out), 1)
            out_neg = F.relu(self.cls1(concat))
            out_neg = self.cls2(out_neg)
            #out_neg = self.sigmoid(out_neg)

        return query_out, support_pos_out, support_neg_out, out_pos, out_neg
