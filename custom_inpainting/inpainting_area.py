from abc import ABC
from copy import deepcopy
import os
from typing import List
import cv2
from custom_inpainting.utils import BBox, open_coco
import random
import numpy as np

from PIL import Image
from pathlib import Path


class InpaintingArea:

    def __init__(self, context_bbox: BBox, inpaint_bbox: BBox) -> None:
        self.context_bbox = context_bbox
        self.inpaint_bbox = inpaint_bbox

    def _get_relative_inpaint_bbox(self) -> BBox:
        return BBox(
            x1 = self.inpaint_bbox.x1 - self.context_bbox.x1,
            y1 = self.inpaint_bbox.y1 - self.context_bbox.y1,
            x2 = self.inpaint_bbox.x2 - self.context_bbox.x1,
            y2 = self.inpaint_bbox.y2 - self.context_bbox.y1,
        )

    def get_inpainting_mask(self):
        mask = np.zeros((self.context_bbox.h, self.context_bbox.w, 1), dtype=np.uint8)
        relative_inpaint_bbox = self._get_relative_inpaint_bbox()
        mask = cv2.rectangle(
            mask, 
            (relative_inpaint_bbox.x1, relative_inpaint_bbox.y1), 
            (relative_inpaint_bbox.x2, relative_inpaint_bbox.y2), 
            (1), 
            -1
        )
        return Image.fromarray(np.uint8(mask[:, :, 0] * 255) , 'L')



class InpaintingAreaMask:

    def __init__(self, context_bbox: BBox, mask: np.ndarray) -> None:
        self.context_bbox = context_bbox
        self.mask = mask * 255

    def get_inpainting_mask(self):
        return Image.fromarray(np.uint8(self.mask[:, :, 0] * 255) , 'L')



class InpaintingAreaGenerator(ABC):

    def __init__(self, context_bbox_size: int,) -> None:
        self.context_bbox_size = context_bbox_size

    def get_inpainting_areas(self, img_path: Path) -> List[InpaintingArea]:
        raise NotImplementedError()


class InpaintingAreaGeneratorCOCO(InpaintingAreaGenerator):

    def __init__(
            self, 
            context_bbox_size: int,
            coco_ann_path: str,
            img_dir: str,
            padding: int = 0
        ): 
        self.context_bbox_size = context_bbox_size
        labeled_images = open_coco(coco_ann_path) # TODO
        self.labeled_images_dict = dict()
        self.padding = padding
        for labeled_image in labeled_images:
            self.labeled_images_dict[labeled_image.name] = deepcopy(labeled_image)

        # Check that all images from the image directory have an image in the coco annotation
        for img_name in os.listdir(img_dir):
            assert img_name in self.labeled_images_dict, f"{img_name} is not in the annotation list"


    def get_inpainting_areas(self, img_path: Path) -> List[InpaintingArea]:
        labeled_image = self.labeled_images_dict[img_path.name] 

        image = Image.open(img_path)
        context_bbox_size = min([self.context_bbox_size, image.height, image.width])
        areas = list()

        for inpaint_bbox in labeled_image.bbox_list:
            
            context_bbox_center_x = min([int(image.width - context_bbox_size / 2), inpaint_bbox.center.x])
            context_bbox_center_y = min([int(image.height - context_bbox_size / 2), inpaint_bbox.center.y])

            context_bbox_x1 = max([0, int(context_bbox_center_x - context_bbox_size / 2)])
            context_bbox_y1 = max([0, int(context_bbox_center_y - context_bbox_size / 2)])

            inpaint_bbox.add_padding(size=self.padding, max_x=image.width, max_y=image.height)

            areas.append(
                InpaintingArea(
                    context_bbox=BBox(
                        x1=context_bbox_x1,
                        y1=context_bbox_y1,
                        x2=context_bbox_x1 + context_bbox_size,
                        y2=context_bbox_y1 + context_bbox_size
                    ),
                    inpaint_bbox=inpaint_bbox
                )
            )

        return areas


class InpaintingAreaGeneratorRandom(InpaintingAreaGenerator):

    def __init__(
            self,   
            context_bbox_size: int,
            inpaint_box_size: int, 
            number_of_areas_per_image: int
        ): 
        self.context_bbox_size = context_bbox_size
        self.inpaint_box_size = inpaint_box_size
        self.number_of_areas_per_image = number_of_areas_per_image

    def get_inpainting_areas(self, img_path: Path) -> List[InpaintingArea]:
        
        areas = list()

        image = Image.open(img_path)
        context_bbox_size = min([self.context_bbox_size, image.height, image.width])
        inpaint_bbox_size = min([context_bbox_size, self.inpaint_box_size])

        for i in range(self.number_of_areas_per_image):
            context_bbox_x1 = random.randint(0, image.width - context_bbox_size)
            context_bbox_y1 = random.randint(0, image.height - context_bbox_size)

            context_bbox_xc = int(context_bbox_x1 + context_bbox_size / 2)
            context_bbox_yc = int(context_bbox_y1 + context_bbox_size / 2)

            inpaint_bbox_x1 = int(context_bbox_xc - inpaint_bbox_size / 2)
            inpaint_bbox_y1 = int(context_bbox_yc - inpaint_bbox_size / 2)

            areas.append(
                InpaintingArea(
                    context_bbox=BBox(
                        x1=context_bbox_x1,
                        y1=context_bbox_y1,
                        x2=context_bbox_x1 + context_bbox_size,
                        y2=context_bbox_y1 + context_bbox_size
                    ),
                    inpaint_bbox=BBox(
                        x1=inpaint_bbox_x1,
                        y1=inpaint_bbox_y1,
                        x2=inpaint_bbox_x1 + inpaint_bbox_size,
                        y2=inpaint_bbox_y1 + inpaint_bbox_size
                    )
                )
            )

        return areas


class InpaintingAreaGeneratorMasks(InpaintingAreaGenerator):
    

    def __init__(self, context_bbox_size: int, masks_dir_path: str) -> None:
        super().__init__(context_bbox_size)

        # receive masks_dir and create dict where key is image base_name and value is mask_path
        self.mask_path_dict = dict()
        for mask_name in os.listdir(masks_dir_path):
            mask_path = os.path.join(masks_dir_path, mask_name)
            self.mask_path_dict[os.path.splitext(os.path.basename(mask_path))[0]] = mask_path




    def get_inpainting_areas(self, img_path: Path) -> List[InpaintingArea]:
        mask_path = self.mask_path_dict[img_path.stem] 

        image = Image.open(img_path)
        context_bbox_size = min([self.context_bbox_size, image.height, image.width])
        areas = list()

        mask = cv2.imread(mask_path)

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if len(cnts) == 0:
            return list()
        
        x,y,w,h = cv2.boundingRect(cnts[0])

        bbox_center_x = int(x + w / 2)
        bbox_center_y = int(y + h / 2)

        context_bbox_center_x = min([int(image.width - context_bbox_size / 2), bbox_center_x])
        context_bbox_center_y = min([int(image.height - context_bbox_size / 2), bbox_center_y])


        context_bbox_x1 = max([0, int(context_bbox_center_x - context_bbox_size / 2)])
        context_bbox_y1 = max([0, int(context_bbox_center_y - context_bbox_size / 2)])
        context_bbox_x2 = context_bbox_x1 + context_bbox_size
        context_bbox_y2 = context_bbox_y1 + context_bbox_size

        cropped_mask = mask[context_bbox_y1:context_bbox_y2, context_bbox_x1:context_bbox_x2] * 255

        areas.append(InpaintingAreaMask(
            context_bbox=BBox(
                x1=context_bbox_x1,
                y1=context_bbox_y1,
                x2=context_bbox_x2,
                y2=context_bbox_y2
            ),
            mask=cropped_mask
        ))

        return areas