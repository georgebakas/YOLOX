from pathlib import Path
import os
import glob
from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
    coco_format,
)
import cv2
import argparse
import json
import numpy as np
import imagesize


# Comment by sandy
#                
# main.py is taken from https://github.com/Taeyoung96/Yolo-to-COCO-format-converter
# and has been modified in order to extract coco from yolo annotation files
# without needing to have a specific images/annoations folder structure
#
# 1. Install dependencies:
#   pip install numpy
#   pip install opencv-python
#   pip install imagesize
#
# 2. Extract coco from yolo:
#   python main.py -im_path images_dir -ann_path annotations_dir --output coco.json
#
#   (creates 'coco.json' with all annotations inside the provided images directory)


#################################################
# Change the classes depend on your own dataset.#
# Don't change the list name 'classes'          #
#################################################

# YOLO_DARKNET_SUB_DIR = "YOLO_darknet"

classes = [
    0,
    1,
    2
]


def get_images_info_and_annotations(opt):
    ann_path = opt.ann_path
    #print('annotation path: ', ann_path)

    path = opt.im_path
    #print('image path: ', path)
    
    annotations = []
    images_annotations = []
    if os.path.isdir(path):
        file_paths = sorted(glob.glob(os.path.join(path,"*.jpg")))
        file_paths += sorted(glob.glob(os.path.join(path,"*.jpeg")))
        file_paths += sorted(glob.glob(os.path.join(path,"*.png")))
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    for file_path in file_paths:
        print('file_path:', file_path)
        # Check how many items have progressed
        print("\rProcessing " + str(image_id) + " ...", end='')

        # Build image annotation, known the image's width and height
        w, h = imagesize.get(str(file_path))
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}.txt"
        print('label_file_name:', label_file_name)
        
        # if opt.yolo_subdir:
        #     annotations_path = file_path.parent / YOLO_DARKNET_SUB_DIR / label_file_name
        #else:
        annotations_path = ann_path + os.sep + label_file_name #file_path.parent / label_file_name
        
        #print("annotations_path:", annotations_path)

        
        
        if not os.path.exists(annotations_path):
            print(f'annotation path {annotations_path} does not exist')
            continue  # The image may not have any applicable annotation txt file.
                
        with open(str(annotations_path), "r") as label_file:
            label_read_line = label_file.readlines()

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line:
            label_line = line1
            category_id = (
                int(label_line.split()[0]) + 1
            )  # you start with annotation id with '1'
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            float_x_center = w * x_center
            float_y_center = h * y_center
            float_width = w * width
            float_height = h * height

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)

        
            annotation = create_annotation_from_yolo_format(
                min_x,
                min_y,
                width,
                height,
                image_id,
                category_id,
                annotation_id,
                segmentation=opt.box2seg,
            )
            annotations.append(annotation)
            annotation_id += 1
        image_id += 1  # if you finished annotation work, updates the image id.

    return images_annotations, annotations


def debug(opt):
    im_path = opt.im_path
    color_list = np.random.randint(low=0, high=256, size=(len(classes), 3)).tolist()

    # read the file
    file = open(im_path, "r")
    read_lines = file.readlines()
    file.close()

    for line in read_lines:
        print("Image Path : ", line)
        # read image file
        img_file = cv2.imread(line[:-1])

        # read .txt file
        label_path = line[:-4] + "txt"
        label_file = open(label_path, "r")
        label_read_line = label_file.readlines()
        label_file.close()

        for line1 in label_read_line:
            label_line = line1

            category_id = label_line.split()[0]
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            int_x_center = int(img_file.shape[1] * x_center)
            int_y_center = int(img_file.shape[0] * y_center)
            int_width = int(img_file.shape[1] * width)
            int_height = int(img_file.shape[0] * height)

            min_x = int_x_center - int_width / 2
            min_y = int_y_center - int_height / 2
            width = int(img_file.shape[1] * width)
            height = int(img_file.shape[0] * height)

            print("class name :", classes[int(category_id)])
            print("x_upper_left : ", min_x, "\t", "y_upper_left : ", min_y)
            print("width : ", width, "\t", "\t", "height : ", height)
            print()

            # Draw bounding box
            cv2.rectangle(
                img_file,
                (int(int_x_center - int_width / 2), int(int_y_center - int_height / 2)),
                (int(int_x_center + int_width / 2), int(int_y_center + int_height / 2)),
                color_list[int(category_id)],
                3,
            )

        cv2.imshow(line, img_file)
        delay = cv2.waitKeyEx()

        # If you press ESC, exit
        if delay == 27 or delay == 113:
            break

        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument(
        "-im",
        "--im_path",
        type=str,
        help="Absolute path for 'train.txt' or 'test.txt', or the root dir for images.",
    )
    
    parser.add_argument(
    "-ann",
    "--ann_path",
    type=str,
    help="Directory path of annotation files.",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Visualize bounding box and print annotation information",
    )
    parser.add_argument(
        "--output",
        default="train_coco.json",
        type=str,
        help="Name the output json file",
    )
    # parser.add_argument(
    #     "--yolo-subdir",
    #     action="store_true",
    #     help="Annotations are stored in a subdir not side by side with images.",
    # )
    parser.add_argument(
        "--box2seg",
        action="store_true",
        help="Coco segmentation will be populated with a polygon "
        "that matches replicates the bounding box data.",
    )
    args = parser.parse_args()
    return args


def main(opt):
    
    output_name = opt.output
    output_path = opt.im_path + os.sep + output_name

    print("Start!")

    if opt.debug is True:
        debug(opt)
        print("Debug Finished!")
    else:
        (
            coco_format["images"],
            coco_format["annotations"],
        ) = get_images_info_and_annotations(opt)

        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        with open(output_path, "w") as outfile:
            json.dump(coco_format, outfile, indent=4)

        print("Finished!")


if __name__ == "__main__":
    options = get_args()
    main(options)
