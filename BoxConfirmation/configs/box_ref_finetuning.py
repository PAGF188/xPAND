from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from .coco_loader import dataloader
from ..data.builtin import register_all_coco
from ..data.builtin import register_all_voc
from detectron2.data import get_detection_dataset_dicts
from ..modeling.rcnn import BoxRefinement
from ..data.dataset_mapper import IoUDatasetMapper
from ..modeling.roi_heads import BoxVerificationStandardROIHeads
from ..modeling.fast_rcnn import BoxVerificationFastRCNNOutputLayers
from ..evaluation.BoxVerificationEvaluation import BoxVerificationEvaluator
import os

### 1) Register datasets. 
register_all_coco(os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets")))
register_all_voc(os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets")))

### 2) Add train and test dataset
dataloader.train.dataset.names = "coco_prueba_finetuning_train_real"
dataloader.test.dataset.names = "coco_prueba_finetuning_test"

### 3) Change datasetmapper to load IoU field to instances
dataloader.train.mapper['_target_'] = IoUDatasetMapper
dataloader.test.mapper['_target_'] = IoUDatasetMapper

### 4) Dataloader modifications
# Add evaluator
dataloader.evaluator['_target_'] = BoxVerificationEvaluator
dataloader['evaluator']['output_dir'] = "./prueba2_finetuning_real/eval/"
dataloader.train.mapper.use_instance_mask = False
dataloader.train.mapper.recompute_boxes = False
dataloader['train'].total_batch_size = 24


### 3) Get model
model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.proposal_generator = None
model['_target_'] = BoxRefinement
model.roi_heads['_target_'] = BoxVerificationStandardROIHeads
model.roi_heads['mask_head'] = False
model.roi_heads['mask_pooler'] = False
model.roi_heads['mask_in_features'] = False
model.roi_heads.box_predictor['_target_'] = BoxVerificationFastRCNNOutputLayers
model.freeze_backbone = True  # To Freeze backbone


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "models_BC/model_final_BC_prueba1.pth"
)
train.eval_period = 10000
train.checkpointer.period = 10000
train.output_dir = "./prueba2_finetuning_real"


train.max_iter = 30000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones= [27000, 29000], #[336417, 364488], #[126157, 136683], #[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.params.base_lr = 0.00001
