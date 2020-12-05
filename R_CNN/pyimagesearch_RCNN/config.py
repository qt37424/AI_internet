#import the necessary packages
import os

#define the base path to the *original* input dataset and then use the base path to derive the image and annotations directories
ORIG_BASE_PATH = "raccoons"      #điền vô đây
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

#define the base path to the *new* dataset after running our dataset builder script and then use the base path to derive the paths
#to our output class label directoris
BASE_PATH = "dataset"
POSITIVE_PATH = os.path.sep.join([BASE_PATH, "raccoon"])#điền vô đây
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoon"]) #điền too

#define the number of max proposals usedd when running selective search for (1) gathering training dât and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

#define the maximum number of positive and negative images to be generated from each page
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

#initialize the input dimensions to the network
INPUT_DIMS = (224, 2244)

#define the path to the output model and label binarizer
MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

#define the minimum probability required for a positive prediction
#(used to filter out false-positive predictions)
MIN_PROBA = 0.99

