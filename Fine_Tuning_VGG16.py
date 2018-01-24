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

def sampling_dataset(train_set, perc_train):
    idx = np.arange(len(train_set))
    len_train = np.int(np.ceil(len(train_set)*perc_train))
    np.random.seed(10)
    train_idx = np.random.choice(idx,len_train, replace= False)
    test_idx = np.setdiff1d(idx, train_idx)

    return(train_idx, test_idx)

train_idx, test_idx = sampling_dataset(train_set = list_train_pics, perc_train = 0.7)

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

train_img = import_images_in_array(list_train_pics = list_train_pics, path = path_source,index_file = train_idx, shape = (256,256))
test_img = import_images_in_array(list_train_pics = list_train_pics,path = path_source, index_file = test_idx , shape = (256,256))
subs_img = import_images_in_array(list_train_pics = subs_images_files, path = subs_images_dir, index_file= [] , shape = (256,256))

#train_img = np.array(train_img)

final_data_img = train_img + test_img
final_labels = label_train + label_test
final_labels_keras = to_categorical(final_labels, 25)
final_data_img = np.array(final_data_img)
final_data_img = final_data_img/255.0

subs_img = np.array(subs_img)
subs_img = subs_img/255.0

#########################################################################################
def create_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape= final_data_img.shape[1:])
    #base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    add_model = Sequential()
    add_model.add(BatchNormalization(input_shape=base_model.output_shape[1:]))
    add_model.add(Flatten())
    add_model.add(Dense(256, activation='relu'))
    add_model.add(BatchNormalization())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(25, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), \
                  metrics=['accuracy'])
    for layer in model.layers[:-1]:
        layer.trainable = False


    return model

def train_model(epochs, train_generator):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=final_data_img.shape[0] // BATCH_SIZE,
        epochs=epochs,verbose = 1
)


BATCH_SIZE = 32
model = create_model()

train_datagen = ImageDataGenerator()
train_datagen.fit(final_data_img)
train_generator = train_datagen.flow(final_data_img, final_labels_keras, batch_size=BATCH_SIZE)

start = time.time()
hist = train_model(10, train_generator)
end = time.time()

##########Saving model to disk#############################################################
model_json = model.to_json()
with open("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/model_VGG16_10epochs.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/model_no_augmentation_all_train_VGG16_10epochs.h5")
print("Saved model to disk")




##############Loading model from Disk###########################################################
json_file = open('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/model_VGG16.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
# load weights into new model
model.load_weights("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/model_no_augmentation_all_train_VGG16.h5")
print("Loaded model from disk")

########Check for the layers#####################

for i, layer in enumerate(model.layers):
    print i ,layer.trainable


for i, layer in enumerate(model.layers):
    print i ,layer

####Freezing the top 15 layers###############
for layer in model.layers[:15]:
   layer.trainable = False
for layer in model.layers[15:]:
   layer.trainable = True

###############################

###Compiling Model###############
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), \
                  metrics=['accuracy'])
#################################################


def train_model(epochs, train_generator):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch= final_data_img.shape[0] // BATCH_SIZE,
        epochs=epochs,verbose = 1
)


BATCH_SIZE = 32

train_datagen = ImageDataGenerator()
train_datagen.fit(final_data_img)
train_generator = train_datagen.flow(final_data_img, final_labels_keras, batch_size=BATCH_SIZE)


start = time.time()
hist = train_model(7, train_generator)
end = time.time()

start = time.time()
preds_subs = model.predict(subs_img, verbose = 1)
end = time.time()

preds_subs_resnet50 = joblib.load('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_subs_Resnet50')

preds_subs_weighted =  0.68 *(preds_subs_resnet50) + 0.59*(preds_subs) ##0.71882####

preds_subs_weighted =  0.75 *(preds_subs_resnet50) + 0.5*(preds_subs)##0.71940#####

preds_subs_weighted = preds_subs_weighted/1.27

preds_subs_weighted = preds_subs_weighted/1.25  ##0.71940#####
preds_subs_final = np.argmax(preds_subs_weighted, axis = 1)

model_json = model.to_json()
with open("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/model_Resnet50_fine_tuned.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/model_no_augmentation_all_train_Resnet50_fine_tuned.h5")
print("Saved model to disk")

joblib.dump(preds_subs,'/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_subs_Resnet50')

preds_subs_final = np.argmax(preds_subs, axis = 1)
preds_subs_final.shape
#np.sum(pred_test_final == np.array(label_test))/(1.0 * len(pred_test_final))


####To be submitted#################
def results_to_be_submitted(test,result_subs , path,name_of_file_sub):
    filename = path+name_of_file_sub+'.csv'
    image_names_subs = [re.search(r'(.*).png',text).group(1) for text in subs_images_files]
    subs_images_pandas = pd.DataFrame(image_names_subs, columns= ['image_id'])

    results_label = [dict_inverse_labels[item] for item in result_subs]
    results_label = pd.DataFrame(results_label, columns= ['label'])
    subs_pd = pd.concat([subs_images_pandas, results_label], axis = 1)
    results_submitted = pd.merge(test, subs_pd , on = 'image_id')
    results_submitted.to_csv(filename,index = False)

########0.68187###################
results_to_be_submitted(test = testx,result_subs = preds_subs_final  ,path = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/Submissions/',
                        name_of_file_sub = "Fine_tuning_cnn_VGG16_Resnet50_diff_weights_10epochs") ##55.6% accuracy


#######################End of Fine Tuning for Resnet50#########################################