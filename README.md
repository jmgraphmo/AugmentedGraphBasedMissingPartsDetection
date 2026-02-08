# AugmentedGraphBasedMissingPartsDetection
This repository contains the adapted code for "Exploring Data Augmentation in Computer Vision for Missing-Object Detection with GNN"

The project is tested in a clean conda environment (02/08/26) and the following commands were used to make it work in 
Visual Studio Code. 



```text

conda create -n thesisjm python=3.11 -y
conda activate thesisjm

conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
conda install -y -c conda-forge numpy pillow pyyaml tqdm
conda install -y -c conda-forge opencv
conda install -y -c conda-forge psutil tabulate scikit-learn

pip install nvidia-ml-py3 torch-geometric

conda install -y -c conda-forge matplotlib

pip install ultralytics

pip install openai
```

All programs have run from the project root !

after cloning the structure should look like this

```text

AugmentedGraphBasedMissingPartsDetection/
├── data/
│   ├── processed/
│   ├── generatedimages/
│   ├── raw/
│   └── splits/
│
├── models/
│   ├── graph_rcnn/
│   └── yolo/
│
├── python/
│   ├── data_cleaning/
│   ├── graph_rcnn/
│   ├── vlm/
│   └── yolo/

```


first download the delftbikes dataset from:

https://data.4tu.nl/articles/dataset/DelftBikes_data_underlying_the_publication_Hallucination_In_Object_Detection-A_Study_In_Visual_Part_Verification/14866116

the zip has a test foler, train folder, and two jsons
extract the content into data.
-delete the test folder and the fake_test_annotation.json
-put the "train_annotations.json" into the data/raw folder
-rename the train folder from train to images

- 

new structure:

```text

AugmentedGraphBasedMissingPartsDetection/
├── data/
│   ├── images/         #all images from the dataset
│   ├── processed/ 
│   ├── generatedimages/
│   ├── raw/
│   └── splits/
│
├── models/
│   ├── graph_rcnn/
│   └── yolo/
│
├── python/
│   ├── data_cleaning/
│   ├── graph_rcnn/
│   ├── vlm/
│   └── yolo/

```


create a copy of the images folder in the same directory called images_vlm

new structure:

```text


AugmentedGraphBasedMissingPartsDetection/
├── data/
│   ├── images/         #all images from the dataset
│   ├── images_vlm/         #all images from the dataset
│   ├── generatedimages/
│   ├── processed/ 
│   ├── raw/
│   └── splits/
│
├── models/
│   ├── graph_rcnn/
│   └── yolo/
│
├── python/
│   ├── data_cleaning/
│   ├── graph_rcnn/
│   ├── vlm/
│   └── yolo/

```


#######             preprocessing                    #######       

run python/data_cleaning/data_processing.py

now you should find final_annotations_without_occluded.json in processed

run python/data_cleaning/json_to_yolo.py

it should create a folder yolo_format in data

#######             augmentation               #######       

python/graph_rcnn and python/yolo contain the models with and without augmentation which can be run now. 

the exception is : graph_rcnn_augmented_zoom_shift_vlm.py



#######             generated data               #######       

vlm/mask_generation.py can be run to create the masks required for the image generation which will be found in data/masks

vlm/image_generation.py can generate the images which will be found in data/generatedimages. An api key for open ai is required .

the other option is to extract the zip inside data/generateimages which contains the images used for the thesis. 

the generated images should be pasted into images_vlm . replace all existing images. 

now it is possible to run graph_rcnn_augmented_zoom_shift_vlm.py as well.

