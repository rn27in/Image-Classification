import numpy as np
import pandas as pd
import cv2,os,time,gc,re
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import shutil
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.externals import joblib
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.regularizers import l2

def load_data(path_to_file):
    train = pd.read_csv(path_to_file + 'train.csv')
    test = pd.read_csv(path_to_file + 'test.csv')

    return(train,test)

trainx,testx = load_data(path_to_file= "../")

path_source = '../train_img/'
list_train_pics = os.listdir(path_source)
subs_images_dir = '../test_img/'
subs_images_files = os.listdir(subs_images_dir)

def dictionaries(train, list_train_pics):
    dict_labels = {j:i for i,j in enumerate(train.label.unique())}
    dict_inverse_labels = {value:key for key,value in dict_labels.iteritems()}
    image_names = [re.search(r'(.*).png',text).group(1) for text in list_train_pics]
    image_names_dict = {row['image_id']:row['label'] for index,row in train.iterrows()}
    label_map = [image_names_dict[image_x] for image_x in image_names]
    label_num = [dict_labels[text] for text in label_map]

    return(dict_labels,dict_inverse_labels,image_names,image_names_dict,label_map,label_num)

dict_labels,dict_inverse_labels,image_names,image_names_dict,label_map,label_num = dictionaries(trainx, list_train_pics)

def sampling_dataset(train_set, perc_train):
    idx = np.arange(len(train_set))
    len_train = np.int(np.ceil(len(train_set)*perc_train))
    np.random.seed(10)
    train_idx = np.random.choice(idx,len_train, replace= False)
    test_idx = np.setdiff1d(idx, train_idx)

    return(train_idx, test_idx)

train_idx, test_idx = sampling_dataset(train_set = list_train_pics, perc_train = 0.7)

####Importing all the models####################################################
model_vgg16 = applications.VGG16(include_top=False, weights='imagenet')
model_resnet = applications.ResNet50(include_top=False, weights='imagenet')
model_xception = applications.Xception(include_top=False, weights='imagenet')
model_inception = applications.InceptionV3(include_top=False, weights='imagenet')
model_vgg19 = applications.VGG19(include_top=False, weights='imagenet')

###Creating an array of images in train and test and doing preprocessing##########

def import_images_in_array(list_train_pics,path,index_file, shape):
    image_list = []
    if len(index_file)> 0:
        for img_idx in index_file:
            image_name = list_train_pics[img_idx]
            temp_img = image.load_img(path + image_name, target_size= shape)
            temp_img = image.img_to_array(temp_img)
            image_list.append(temp_img)
        image_list = np.array(image_list)
        image_list = image_list/255.0
    elif len(index_file)== 0:
        for image_item in list_train_pics:
            temp_img = image.load_img(path + image_item, target_size= shape)
            temp_img = image.img_to_array(temp_img)
            image_list.append(temp_img)
        image_list = np.array(image_list)
        image_list = image_list / 255.0

    return(image_list)

train_img = import_images_in_array(list_train_pics = list_train_pics, path = path_source,index_file = train_idx, shape = (224,224))
test_img = import_images_in_array(list_train_pics = list_train_pics,path = path_source, index_file = test_idx , shape = (224,224))
subs_img = import_images_in_array(list_train_pics = subs_images_files, path = subs_images_dir, index_file= [] , shape = (224,224))


def labels_for_model(label,train_idx, test_idx):
    label_train = [label[index] for index in train_idx]
    label_test = [label[index] for index in test_idx]
    label_train_keras = to_categorical(label_train)
    label_test_keras = to_categorical(label_test)

    return(label_train,label_test,label_train_keras,label_test_keras)

label_train,label_test,label_train_keras,label_test_keras = labels_for_model(label = label_num,
train_idx = train_idx,test_idx = test_idx)



############Extracting features and dumping to disk###############
def extract_features(model, image_array,path,name_of_data , shape, to_be_saved = False):
    features = model.predict(image_array)
    if to_be_saved == True:
        joblib.dump(features, path+name_of_data)

#######Extracting features from all images################
##This has to be done for training and testing images###############
start = time.time()
train_features_resnet = extract_features(model = model_vgg19, image_array= subs_img,
path='../VGG19/',
name_of_data= 'subs_features_VGG19', shape = (224,224,3), to_be_saved= True)
end = time.time()

######################################End of pre-processing##############################



