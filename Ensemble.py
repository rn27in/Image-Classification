from sklearn.externals import joblib
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
from keras.models import model_from_json

def load_data(path_to_file):
    train = pd.read_csv(path_to_file + 'train.csv')
    test = pd.read_csv(path_to_file + 'test.csv')

    return(train,test)

trainx,testx = load_data(path_to_file= "/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/")

path_source = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/train_img/'
list_train_pics = os.listdir(path_source)
subs_images_dir = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/test_img/'
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

##############################################################################################

def labels_for_model(label,train_idx, test_idx):
    label_train = [label[index] for index in train_idx]
    label_test = [label[index] for index in test_idx]
    label_train_keras = to_categorical(label_train)
    label_test_keras = to_categorical(label_test)

    return(label_train,label_test,label_train_keras,label_test_keras)

label_train,label_test,label_train_keras,label_test_keras = labels_for_model(label = label_num,
train_idx = train_idx,test_idx = test_idx)


###############Importing Images#######################################################
def import_images_in_array(list_train_pics,path,index_file, shape):
    image_list = []
    if len(index_file)> 0:
        for img_idx in index_file:
            image_name = list_train_pics[img_idx]
            temp_img = image.load_img(path + image_name, target_size= shape)
            temp_img = image.img_to_array(temp_img)
            image_list.append(temp_img)
        #image_list = np.array(image_list)
        #image_list = image_list/255.0
    elif len(index_file)== 0:
        for image_item in list_train_pics:
            temp_img = image.load_img(path + image_item, target_size= shape)
            temp_img = image.img_to_array(temp_img)
            image_list.append(temp_img)
        #image_list = np.array(image_list)
        #image_list = image_list / 255.0

    return(image_list)

subs_img = import_images_in_array(list_train_pics = subs_images_files, path = subs_images_dir, index_file= [] , shape = (256,256))
subs_img = np.array(subs_img)
subs_img = subs_img/255.0


pred_resnet50 = joblib.load('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_subs_Resnet50')
pred_xception = joblib.load('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/preds_subs_final_xception')
pred_inception = joblib.load('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_test_xception')
pred_vgg16 = joblib.load('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_subs_Resnet50')
pred_vgg19 = joblib.load('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_subs_Resnet50')


list_models = ['Resnet50', 'InceptionV3', 'VGG16', 'VGG19', 'Xception']
path = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/'
path_to_dump = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/Preds/'
start = time.time()
for elem in list_models:
    print elem
    json_path =  path+'model_'+elem+'_10epochs.json'
    weights_path = path + 'model_no_augmentation_all_train_'+elem + '_10epochs.h5'
    json_file = open(json_path, 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    # load weights into new model
    model.load_weights(weights_path)
    print("Loaded model from disk")
    preds_subs = model.predict(subs_img, verbose=1)
    joblib.dump(preds_subs , path_to_dump+'preds_final_'+elem)
    del model
    gc.collect()
end = time.time()

####################################################
preds_subs = []
for elem in list_models:
    preds_subs.append(joblib.load(path_to_dump+'preds_final_'+ elem))

['preds_subs_'+elem for elem in list_models]

preds_subs_Resnet50, preds_subs_InceptionV3, preds_subs_VGG16, preds_subs_VGG19, preds_subs_Xception = preds_subs

preds_final = (0.683)*preds_subs_Resnet50+ (0.6691)*preds_subs_InceptionV3+ (0.59)* preds_subs_VGG16+ (0.62)*preds_subs_VGG19+ (0.788)*preds_subs_Xception

preds_final = preds_final/(0.683+0.6691+0.59+0.62+0.788)  ##0.82390


preds_final = (4)*preds_subs_Resnet50+ (3)*preds_subs_InceptionV3+ (1)* preds_subs_VGG16+ (2)*preds_subs_VGG19+ (5)*preds_subs_Xception

preds_final = preds_final/(15)  ##0.82621



preds_subs_final = np.argmax(preds_final, axis = 1)


def results_to_be_submitted(test,result_subs , path,name_of_file_sub):
    filename = path+name_of_file_sub+'.csv'
    image_names_subs = [re.search(r'(.*).png',text).group(1) for text in subs_images_files]
    subs_images_pandas = pd.DataFrame(image_names_subs, columns= ['image_id'])

    results_label = [dict_inverse_labels[item] for item in result_subs]
    results_label = pd.DataFrame(results_label, columns= ['label'])
    subs_pd = pd.concat([subs_images_pandas, results_label], axis = 1)
    results_submitted = pd.merge(test, subs_pd , on = 'image_id')
    results_submitted.to_csv(filename,index = False)

results_to_be_submitted(test = testx,result_subs = preds_subs_final  ,path = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/Submissions/',
                        name_of_file_sub = "results_weighted_avg_more_weights_ensemble") ##55.6% accuracy



preds_inception = joblib.load(path_to_dump'preds_final_InceptionV3