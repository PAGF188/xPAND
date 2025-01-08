

import os
import datetime
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode
import json
import argparse

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

def load_filtered_voc_instances(
    name: str, dirname: str, split: str, classnames: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join(os.path.split(dirname)[0], "vocsplit")
        shot = name.split("_")[-2].split("shot")[0]
        seed = int(name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed{}".format(seed))
        for cls in classnames:
            with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                fileids[cls] = fileids_
    else:
        with PathManager.open(
            os.path.join(dirname, "ImageSets", "Main", split + ".txt")
        ) as f:
            fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join(os.path.split(dirname)[0], "VOC{}".format(year))
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname, "JPEGImages", fileid + ".jpg"
                )

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts






def convert_to_coco_dict(dataset_dicts, categories):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.
    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset
    COCO data format description can be found here:
    http://cocodataset.org/#format-data
    Args:
        dataset_dicts (list[dict]): a list of dicts following the standard
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(categories)
    ]

    print("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            #"file_name": str(image_dict["file_name"]),  # Modificado para obtener solo el nombre final (y no el directorio) !
            "file_name": str(os.path.basename(image_dict["file_name"])),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    print(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict



def _get_voc_fewshot_instances_meta():
    ret = {
        "thing_classes": PASCAL_VOC_ALL_CATEGORIES,
        "novel_classes": PASCAL_VOC_NOVEL_CATEGORIES,
        "base_classes": PASCAL_VOC_BASE_CATEGORIES,
    }
    return ret

METASPLITS = [
    ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
    ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
    ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
    ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
    ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
    ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
    ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
    ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
    ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
    ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
    ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
    ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
    ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
    ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
    ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
    ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
    ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
    ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
    ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
    ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
    ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
]

for prefix in ["all", "novel"]:
    for sid in range(1, 4):
        for shot in [1, 2, 3, 5, 10]:
            for year in [2007, 2012]:
                for seed in range(30):
                    seed = "_seed{}".format(seed)
                    name = "voc_{}_trainval_{}{}_{}shot{}".format(
                        year, prefix, sid, shot, seed
                    )
                    dirname = "VOC{}".format(year)
                    img_file = "{}_{}shot_split_{}_trainval".format(
                        prefix, shot, sid
                    )
                    keepclasses = (
                        "base_novel_{}".format(sid)
                        if prefix == "all"
                        else "novel{}".format(sid)
                    )
                    METASPLITS.append(
                        (name, dirname, img_file, keepclasses, sid)
                    )

def main(args):
    for name, dirname, split, keepclasses, sid in METASPLITS:
        metadata = _get_voc_fewshot_instances_meta()

        if keepclasses.startswith("base_novel"):
            thing_classes = metadata["thing_classes"][sid]
        elif keepclasses.startswith("base"):
            thing_classes = metadata["base_classes"][sid]
        elif keepclasses.startswith("novel"):
            thing_classes = metadata["novel_classes"][sid]
        
        dirname = os.path.join(args.path, dirname)

        dicts = load_filtered_voc_instances(name, dirname, split, thing_classes)
        print("Loaded {} dicts for {}".format(len(dicts), name))
        coco_format = convert_to_coco_dict(dicts, thing_classes)
        print("Converted to COCO format. Annotations {} Images {}".format(len(coco_format["annotations"]), len(coco_format["images"])))
        print("###############", dirname, name)
        print("First dict: ", dicts[0])
        
        # Save result
        save_name = os.path.join(args.output, f"{name}.json")
        with open(save_name, 'w') as f:
            json.dump(coco_format, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare voc dataset')
    parser.add_argument('--output', required=True, default="/datasets/voc_coco/")
    parser.add_argument('--path', required=True, default="/datasets/")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)