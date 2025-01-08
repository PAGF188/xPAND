from torch.utils.data import Dataset
import json
import torch
import random
import copy
import os
from PIL import Image
import numpy as np
from tabulate import tabulate

def crop(img, bb):
    # bb -> XYXY format
    if (bb[0] < bb[2]) and (bb[1] < bb[3]):
        crop = img[bb[1]:bb[3], bb[0]:bb[2], :]

    return crop


def obtain_images_dict(images):
    images_dict = {}
    for k in images:
        images_dict[k['id']] = k['file_name']

    return images_dict


def obtain_support_dict(support_dict_annotations, clases):
    support_grouped_class = {key:[] for key in clases}

    for sup in support_dict_annotations:
        support_grouped_class[sup['category_id']].append(sup)

    return support_grouped_class


def obtain_support_instance(positive_supports_anots, images, dtpath, transforms, image_format, augmentations_support=None):
    """
    """
    positive_supports_crops = []
    for ps in positive_supports_anots:
        image_name = images[ps['image_id']]
        image_path = os.path.join(dtpath, image_name)

        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            image = np.asarray(image)
            if image_format == 'BGR':
                image = image[:, :, ::-1]
            # # DEBUG! TO SEE IMAGE
            # im = Image.fromarray(image)
            # im.save("./image.png")

        x1, y1, w, h = ps['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
        object_crop = crop(image, [x1,y1,x2,y2])
        object_crop = torch.tensor(transforms(object_crop)['pixel_values'][0])

        ## Add augmentation
        if augmentations_support is not None:
            object_crop = augmentations_support(object_crop)

        positive_supports_crops.append(object_crop)

    positive_supports_crops = torch.stack(positive_supports_crops)
    
    return positive_supports_crops


def compute_dataset_statics(dict_annotations, clases):
    object_dict = {x:0 for x in clases.keys()}
    images = []

    for obj in dict_annotations:
        object_dict[obj['category_id']] += 1
        images.append(obj['image_id'])

    images = set(images)

    # Transform object_dict from classes id to category names
    object_dict_names = {clases[k]:v for k,v in object_dict.items()}
    
    # Print results
    print(f"Number of images: {len(images)}")
    print(tabulate(object_dict_names.items(), headers = ['category', '#instances']))


def obtain_classes_names(categories, clases_id):
    category_name_id = {x:None for x in clases_id}
    for cat in categories:
        if cat['id'] in clases_id:
            category_name_id[cat['id']] = cat['name']
    
    return category_name_id


class COCODatasetMetaLearning(Dataset):
    """
    Coco format dataset class for meta-learning

    TO DO:

    Parameters
    ----------
    json_train : str
        Path to the train data. Coco format must be.
    json_support: str
        Path to support data. Coco format
    dtpath: str
        Path to the images
    transforms: transformers.AutoImageProcessor (hugginface)
        Only includes resize, normalization, ...
    image_format: str
        Image format (BGR, RGB) that will have the returned images
    augmentations_query: torchvision.transforms
        More complex augmentations for query images (mosaic, jitter,...)
    augmentations_support: torchvision.transforms
        More complex augmentations for support images (mosaic, jitter,...)
    support_size: int
        Number of support examples. In >1, the mean is considered per net run
    seed: int

    Returns
    -------

    """
    def __init__(self, json_train, json_support, dtpath, transforms, image_format="BGR", augmentations_query=None, augmentations_support=None, support_size=1, seed=88):
        random.seed(seed)
        with open(json_train) as train:
            train_dict = json.load(train)

        with open(json_support) as support:
            support_dict = json.load(support)
        
        # Check label consistancy
        et1 = set([x['category_id'] for x in train_dict['annotations']])
        et2 = set([x['category_id'] for x in support_dict['annotations']])
        assert et1 == et2
        self.classes = list(et1)
        self.classes_id_to_names_dict = obtain_classes_names(train_dict['categories'], self.classes)
        self.transforms = transforms
        self.augmentations_query = augmentations_query
        self.augmentations_support = augmentations_support
        self.dtpath = dtpath
        self.support_size = support_size
        self.image_format = image_format

        # Save object anotations. FILTER IS CROWD
        self.train_dict_annotations = [x for x in train_dict['annotations'] if x['iscrowd']==0]

        # Filter ignore region if necesary
        if 'ignore_qe' in self.train_dict_annotations[0].keys(): 
            self.train_dict_annotations = [x for x in self.train_dict_annotations if x['ignore_qe']==0]

        # Filter 0 box annotations
        self.train_dict_annotations = [x for x in self.train_dict_annotations if x['bbox'][2]>2 and x['bbox'][3]>2]


        # Save images in dict by id
        self.train_dict_images = obtain_images_dict(train_dict['images'])
        
        # Save support information. Support objects grouped by class
        # Filtering
        aux_supp = [x for x in support_dict['annotations'] if x['iscrowd']==0]
        # Filter ignore region if necesary
        if 'ignore_qe' in aux_supp[0].keys(): 
            aux_supp = [x for x in aux_supp if x['ignore_qe']==0]
        # Filter 0 box annotations
        aux_supp = [x for x in aux_supp if x['bbox'][2]>2 and x['bbox'][3]>2]
        self.support_dict_annotations = obtain_support_dict(aux_supp, self.classes)
        self.support_dict_images = obtain_images_dict(support_dict['images'])

        # Print results
        print(f"Class assign: {self.classes_id_to_names_dict}")
        print(f"Train dataset: {json_train}")
        compute_dataset_statics(self.train_dict_annotations, self.classes_id_to_names_dict)

        print(f"Support dataset: {json_support}")
        compute_dataset_statics(support_dict['annotations'], self.classes_id_to_names_dict)

    def get_classes(self):
        return self.classes

    def __len__(self):
        # len() is the number of objects
        return len(self.train_dict_annotations)


    def __getitem__(self, idx):
        # Get annotation
        anot = self.train_dict_annotations[idx]

        # Get image path
        image_name = self.train_dict_images[anot['image_id']]
        image_path = os.path.join(self.dtpath, image_name)

        ### Open image PIL -> RGB format
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            image = np.asarray(image)
            if self.image_format == 'BGR':
                image = image[:, :, ::-1]
            # # DEBUG! TO SEE IMAGE
            # im = Image.fromarray(image)
            # im.save("./image.png")

        ### Get object crop. The input bbox is in XYHW format
        x1, y1, w, h = anot['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
        object_crop = crop(image, [x1,y1,x2,y2])

        # # DEBUG! TO SEE CROP
        # im = Image.fromarray(object_crop)
        # im.save("./crop.png")
        object_crop = torch.tensor(self.transforms(object_crop)['pixel_values'][0])
        
        # Apply augmentations
        object_crop = self.augmentations_query(object_crop)

        ## DEBUG!
        # import matplotlib.pyplot as plt
        # plt.imshow(object_crop.permute(1, 2, 0))
        # plt.savefig("./crop.png")


        # Encapsulate information in dict
        object_crop_instance = {"object_crop": object_crop, "category_id": anot["category_id"]}

        ### Get negative support class (random)
        c_aux = copy.deepcopy(self.classes)
        c_aux.remove(object_crop_instance['category_id'])
        class_neg = random.choice(c_aux)

        # # The support number is the smallest between the support shot argument, and the support instances of positive/negatives classes
        k_shot = np.minimum(np.minimum(self.support_size, len(self.support_dict_annotations[object_crop_instance['category_id']])), np.minimum(self.support_size, len(self.support_dict_annotations[class_neg])))

        ### Get positive support (random)
        positive_supports = random.choices(self.support_dict_annotations[object_crop_instance['category_id']], k=k_shot)
        positive_support_crop = obtain_support_instance(positive_supports, self.support_dict_images, self.dtpath, self.transforms, self.image_format, self.augmentations_support)
        positive_support_instance = {"object_crop": positive_support_crop, "category_id": anot["category_id"]}

        ### Get negative support (random)
        negative_supports = random.choices(self.support_dict_annotations[class_neg], k=k_shot)
        negative_support_crop = obtain_support_instance(negative_supports, self.support_dict_images, self.dtpath, self.transforms, self.image_format, self.augmentations_support)
        negative_support_instance = {"object_crop": negative_support_crop, "category_id": class_neg}

        return object_crop_instance, positive_support_instance, negative_support_instance
    




class COCODatasetMetaLearningTEST(Dataset):
    """
    Coco format dataset class for meta-learning. Only for test

    TO DO:

    Parameters
    ----------
    json_train : str
        Path to the train data. Coco format must be.
    json_support: str
        Path to support data. Coco format
    dtpath: str
        Path to the images
    transforms: transformers.AutoImageProcessor (hugginface)
        Only includes resize, normalization, ...
    image_format: str
        Image format (BGR, RGB) that will have the returned images
    support_size: int
        Number of support examples. In >1, the mean is considered per net run
    seed: int

    Returns
    -------

    """
    def __init__(self, json_test, json_support, dtpath, transforms, image_format="BGR", support_size=1, seed=88):
        random.seed(seed)
        with open(json_test) as train:
            train_dict = json.load(train)

        with open(json_support) as support:
            support_dict = json.load(support)

        # Check label consistancy
        et1 = set([x['category_id'] for x in train_dict['annotations']])
        et2 = set([x['category_id'] for x in support_dict['annotations']])
        assert et1 == et2
        self.classes = list(et1)
        self.classes_id_to_names_dict = obtain_classes_names(train_dict['categories'], self.classes)
        self.transforms = transforms
        self.dtpath = dtpath
        self.support_size = support_size
        self.image_format = image_format

        # Save object anotations. FILTER IS CROWD
        self.train_dict_annotations = [x for x in train_dict['annotations'] if x['iscrowd']==0]

        # Filter ignore region if necesary
        if 'ignore_qe' in self.train_dict_annotations[0].keys(): 
            self.train_dict_annotations = [x for x in self.train_dict_annotations if x['ignore_qe']==0]
        
        # Filter 0 box annotations
        self.train_dict_annotations = [x for x in self.train_dict_annotations if x['bbox'][2]>2 and x['bbox'][3]>2]
        
        # Save images in dict by id
        self.train_dict_images = obtain_images_dict(train_dict['images'])
        

        # Save support information. Support objects grouped by class
        # Filtering
        aux_supp = [x for x in support_dict['annotations'] if x['iscrowd']==0]
        # Filter ignore region if necesary
        if 'ignore_qe' in aux_supp[0].keys(): 
            aux_supp = [x for x in aux_supp if x['ignore_qe']==0]
        # Filter 0 box annotations
        aux_supp = [x for x in aux_supp if x['bbox'][2]>2 and x['bbox'][3]>2]
        self.support_dict_annotations = obtain_support_dict(aux_supp, self.classes)
        self.support_dict_images = obtain_images_dict(support_dict['images'])


        # Print results
        print(f"Class assign: {self.classes_id_to_names_dict}")
        print(f"Test dataset: {json_test}")
        compute_dataset_statics(self.train_dict_annotations, self.classes_id_to_names_dict)

        print(f"Support dataset Test: {json_support}")
        compute_dataset_statics(support_dict['annotations'], self.classes_id_to_names_dict)

    def get_classes(self):
        return self.classes

    def __len__(self):
        # len() is the number of objects
        return len(self.train_dict_annotations)


    def __getitem__(self, idx):
        # Get annotation
        anot = self.train_dict_annotations[idx]

        # Get image path
        image_name = self.train_dict_images[anot['image_id']]
        image_path = os.path.join(self.dtpath, image_name)

        ### Open image PIL -> RGB format
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            image = np.asarray(image)
            if self.image_format == 'BGR':
                image = image[:, :, ::-1]
            # # DEBUG! TO SEE IMAGE
            # im = Image.fromarray(image)
            # im.save("./image.png")

        ### Get object crop. The input bbox is in XYHW format
        x1, y1, w, h = anot['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
        object_crop = crop(image, [x1,y1,x2,y2])

        # # DEBUG! TO SEE CROP
        # im = Image.fromarray(object_crop)
        # im.save("./crop.png")
        object_crop = torch.tensor(self.transforms(object_crop)['pixel_values'][0])
        
        # Encapsulate information in dict
        object_crop_instance = {"object_crop": object_crop, "category_id": anot["category_id"]}


        ### Get negative support class (random)
        c_aux = copy.deepcopy(self.classes)
        c_aux.remove(object_crop_instance['category_id'])
        class_neg = random.choice(c_aux)

        # The support number is the smallest between the support shot argument, and the support instances of positive/negatives classes
        k_shot = np.minimum(np.minimum(self.support_size, len(self.support_dict_annotations[object_crop_instance['category_id']])), np.minimum(self.support_size, len(self.support_dict_annotations[class_neg])))

        ### Get positive support. 
        positive_supports = random.choices(self.support_dict_annotations[object_crop_instance['category_id']], k=k_shot)
        positive_support_crop = obtain_support_instance(positive_supports, self.support_dict_images, self.dtpath, self.transforms, self.image_format)
        positive_support_instance = {"object_crop": positive_support_crop, "category_id": anot["category_id"]}

        ### Get negative support. 
        negative_supports = random.choices(self.support_dict_annotations[class_neg], k=k_shot)
        negative_support_crop = obtain_support_instance(negative_supports, self.support_dict_images, self.dtpath, self.transforms, self.image_format)
        negative_support_instance = {"object_crop": negative_support_crop, "category_id": class_neg}

        
        return object_crop_instance, positive_support_instance, negative_support_instance



class COCOPseudoLabelMining(Dataset):
    """
    Coco format dataset class for apply the label confirmation step.


    Parameters
    ----------
    pseudos_json : str
        Path to pseudo-annotations. Coco format must be.
    precomputed_support: str
        Path to support data. Coco format
    dtpath: str
        Path to the images
    transforms: transformers.AutoImageProcessor (hugginface)
        Only includes resize, normalization, ...
    image_format: str
        Image format (BGR, RGB) that will have the returned images
    seed: int

    Returns
    -------
    """
    def __init__(self, json_pseudos, precomputed_support, dtpath, transforms, image_format="BGR", seed=88):
        random.seed(seed)
        with open(json_pseudos) as pseudos:
            pseudos_dict = json.load(pseudos)

        # Check label consistancy
        et1 = set([x['category_id'] for x in pseudos_dict['annotations']])
        et2 = set(precomputed_support.keys())
        assert et1.issubset(et2) or et1==et2
        self.classes = list(et2)
        self.classes_id_to_names_dict = obtain_classes_names(pseudos_dict['categories'], self.classes)
        self.transforms = transforms
        self.dtpath = dtpath
        self.image_format = image_format

        self.pseudos_dict_annotations = pseudos_dict['annotations']

        # Filter ignore region if necesary
        if 'ignore_qe' in pseudos_dict['annotations'][0].keys(): 
            self.pseudos_dict_annotations = [x for x in self.pseudos_dict_annotations if x['ignore_qe']==0]
        
        # Filter 0 box annotations
        self.pseudos_dict_annotations = [x for x in self.pseudos_dict_annotations if x['bbox'][2]>2 and x['bbox'][3]>2]
        
        # Save images in dict by id
        self.train_dict_images = obtain_images_dict(pseudos_dict['images'])
        

        # Save support information. Support objects grouped by class
        # Filter 0 box annotations
        self.precomputed_support = precomputed_support


        # Print results
        print(f"Class assign: {self.classes_id_to_names_dict}")
        print(f"Pseudo dataset: {json_pseudos}")
        compute_dataset_statics(self.pseudos_dict_annotations, self.classes_id_to_names_dict)


    def get_classes(self):
        return self.classes

    def __len__(self):
        # len() is the number of objects
        return len(self.pseudos_dict_annotations)
    
    def __getitem__(self, idx):
        # Get annotation
        anot = self.pseudos_dict_annotations[idx]

        # Get image path
        image_name = self.train_dict_images[anot['image_id']]
        image_path = os.path.join(self.dtpath, image_name)

        ### Open image PIL -> RGB format
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            image = np.asarray(image)
            if self.image_format == 'BGR':
                image = image[:, :, ::-1]
            # # DEBUG! TO SEE IMAGE
            # im = Image.fromarray(image)
            # im.save("./image.png")

        ### Get object crop. The input bbox is in XYHW format
        x1, y1, w, h = anot['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
        object_crop = crop(image, [x1,y1,x2,y2])

        # # DEBUG! TO SEE CROP
        # im = Image.fromarray(object_crop)
        # im.save("./crop.png")
        object_crop = torch.tensor(self.transforms(object_crop)['pixel_values'][0])
        
        # Encapsulate information in dict
        object_crop_instance = {"object_crop": object_crop, "category_id": anot["category_id"], "id": anot['id']}

        ### Get positive support. 
        positive_support_instance = {"object_crop": self.precomputed_support[anot["category_id"]], "category_id": anot["category_id"]}


        return object_crop_instance, positive_support_instance
    


class COCOSupportDataset(Dataset):
    # PASCAL VOC categories
    PASCAL_VOC_ALL_CATEGORIES = {
        1: ["aeroplane", "bicycle", "boat", "bottle", "car",
            "cat", "chair", "diningtable", "dog", "horse",
            "person", "pottedplant", "sheep", "train", "tvmonitor",
            "bird", "bus", "cow", "motorbike", "sofa",
        ],
        2: ["bicycle", "bird", "boat", "bus", "car",
            "cat", "chair", "diningtable", "dog", "motorbike",
            "person", "pottedplant", "sheep", "train", "tvmonitor",
            "aeroplane", "bottle", "cow", "horse", "sofa",
        ],
        3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
            "car", "chair", "cow", "diningtable", "dog",
            "horse", "person", "pottedplant", "train", "tvmonitor",
            "boat", "cat", "motorbike", "sheep", "sofa",
        ],
    }

    PASCAL_VOC_NOVEL_CATEGORIES = {
        1: ["bird", "bus", "cow", "motorbike", "sofa"],
        2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
        3: ["boat", "cat", "motorbike", "sheep", "sofa"],
    }

    PASCAL_VOC_BASE_CATEGORIES = {
        1: ["aeroplane", "bicycle", "boat", "bottle", "car",
            "cat", "chair", "diningtable", "dog", "horse",
            "person", "pottedplant", "sheep", "train", "tvmonitor",
        ],
        2: ["bicycle", "bird", "boat", "bus", "car",
            "cat", "chair", "diningtable", "dog", "motorbike",
            "person", "pottedplant", "sheep", "train", "tvmonitor",
        ],
        3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
            "car", "chair", "cow", "diningtable", "dog",
            "horse", "person", "pottedplant", "train", "tvmonitor",
        ],
    }



    """
    Coco format dataset class for precompute the support


    Parameters
    ----------
    json_pseudos : str
        Path to the json data. Coco format must be.
    dtpath: str
        Path to the images
    transforms: transformers.AutoImageProcessor (hugginface)
        Only includes resize, normalization, ...
    image_format: str
        Image format (BGR, RGB) that will have the returned images
    seed: int

    Returns
    -------
    """

    def __init__(self, json_support, dtpath, transforms, image_format="BGR", seed=88):
        random.seed(seed)

        with open(json_support) as support:
            support_dict = json.load(support)

        # Only for VOC
        # Build mapper from continuous id 0,1,2,3,4 to 16,17,18,19. 
        if "voc" in os.path.basename(json_support):
            cats = list(set([x['category_id'] for x in support_dict['annotations']]))
            real_ids_ = [15,16,17,18,19] 
            mapper = {k:j for k,j in zip(cats, real_ids_)}

            for anot in support_dict['annotations']:
                anot['category_id'] = mapper[anot['category_id']]

            for cat_ in support_dict['categories']:
                cat_ ['id'] =  mapper[cat_['id']]


        # Check label consistancy
        et2 = set([x['category_id'] for x in support_dict['annotations']])
        self.classes = list(et2)
        self.classes_id_to_names_dict = obtain_classes_names(support_dict['categories'], self.classes)
        self.transforms = transforms
        self.dtpath = dtpath
        self.image_format = image_format        
        

        # Save support information. Support objects grouped by class
        # Filter 0 box annotations
        aux_supp = [x for x in support_dict['annotations'] if x['bbox'][2]>2 and x['bbox'][3]>2]
        self.support_dict_annotations = obtain_support_dict(aux_supp, self.classes)
        self.support_dict_images = obtain_images_dict(support_dict['images'])


        # Print results
        print(f"Support dataset Test: {json_support}")
        compute_dataset_statics(support_dict['annotations'], self.classes_id_to_names_dict)

    def get_classes(self):
        return self.classes    

    def __len__(self):
        # len() is the number of objects
        return len(self.support_dict_annotations.keys())
    

    def __getitem__(self, idx):
        
        support_class = list(self.support_dict_annotations.keys())[idx]
        support_anots = self.support_dict_annotations[support_class]

        support_tensor = obtain_support_instance(support_anots, self.support_dict_images, self.dtpath, self.transforms, self.image_format)
        support_instance = {"object_crop": support_tensor, "category_id": support_class}
        return support_instance
















class COCODatasetBase(Dataset):
    """
    Coco format dataset class for similarity distance


    Parameters
    ----------
    json_train : str
        Path to the train data. Coco format must be.
    dtpath: str
        Path to the images
    transforms: transformers.AutoImageProcessor (hugginface)
        Only includes resize, normalization, ...
    image_format: str
        Image format (BGR, RGB) that will have the returned images
    augmentations_query: torchvision.transforms
        More complex augmentations for query images (mosaic, jitter,...)
    seed: int

    Returns
    -------
    """

    def __init__(self, json_train, dtpath, transforms, image_format="BGR", augmentations_query=None, seed=88):
        random.seed(seed)
        with open(json_train) as train:
            train_dict = json.load(train)


        # Check label consistancy
        et1 = set([x['category_id'] for x in train_dict['annotations']])
        self.classes = list(et1)
        self.classes_id_to_names_dict = obtain_classes_names(train_dict['categories'], self.classes)
        self.transforms = transforms
        self.augmentations_query = augmentations_query
        self.dtpath = dtpath
        self.image_format = image_format
        # Save object anotations. FILTER IS CROWD.
        self.train_dict_annotations = [x for x in train_dict['annotations'] if x['iscrowd']==0]

        # Filter ignore region if necesary
        if 'ignore_qe' in self.train_dict_annotations[0].keys(): 
            self.train_dict_annotations = [x for x in self.train_dict_annotations if x['ignore_qe']==0]

        # Filter 0 box annotations
        self.train_dict_annotations = [x for x in self.train_dict_annotations if x['bbox'][2]>2 and x['bbox'][3]>2]

        # Save images in dict by id
        self.train_dict_images = obtain_images_dict(train_dict['images'])

        # Print results
        print(f"Class assign: {self.classes_id_to_names_dict}")
        print(f"Train dataset: {json_train}")
        compute_dataset_statics(self.train_dict_annotations, self.classes_id_to_names_dict)


    def get_classes(self):
        return self.classes    

    def __len__(self):
        # len() is the number of objects
        return len(self.train_dict_annotations)


    def __getitem__(self, idx):
        # Get annotation
        anot = self.train_dict_annotations[idx]

        # Get image path
        image_name = self.train_dict_images[anot['image_id']]
        image_path = os.path.join(self.dtpath, image_name)

        ### Open image PIL -> RGB format
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            image = np.asarray(image)
            if self.image_format == 'BGR':
                image = image[:, :, ::-1]
            # # DEBUG! TO SEE IMAGE
            # im = Image.fromarray(image)
            # im.save("./image.png")

        ### Get object crop. The input bbox is in XYHW format
        x1, y1, w, h = anot['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
        object_crop = crop(image, [x1,y1,x2,y2])

        # # DEBUG! TO SEE CROP
        # im = Image.fromarray(object_crop)
        # im.save("./crop.png")
        object_crop = torch.tensor(self.transforms(object_crop)['pixel_values'][0])
        

        # Encapsulate information in dict
        object_crop_instance = {"object_crop": object_crop, "category_id": anot["category_id"], "anot_id": anot['id']}

        return object_crop_instance