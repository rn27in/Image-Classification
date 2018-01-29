import numpy as np
import pandas as pd
import cv2,os,time,gc,re
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential, Input
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

def create_model(base_model):

    # get layers and add average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # add fully-connected layer
    x = Dense(256, activation='relu')(x)

    # add output layer
    predictions = Dense(25, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in base_model.layers:
        layer.trainable = False

    # update the weight that are added
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9), metrics=['accuracy'])

    return model

base_model = Xception(input_shape= final_data_img.shape[1:], weights='imagenet', include_top=False)

model = create_model(base_model)

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

model_json = model.to_json()
with open("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/model_Xception_10epochs.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/model_no_augmentation_all_train_Xception_10epochs.h5")
print("Saved model to disk")

start = time.time()
pred_test = model.predict(subs_img, verbose= 1)
end = time.time()

joblib.dump(pred_test, '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/pred_test_xception')

def results_to_be_submitted(test,result_subs , path,name_of_file_sub):
    filename = path+name_of_file_sub+'.csv'
    image_names_subs = [re.search(r'(.*).png',text).group(1) for text in subs_images_files]
    subs_images_pandas = pd.DataFrame(image_names_subs, columns= ['image_id'])

    results_label = [dict_inverse_labels[item] for item in result_subs]
    results_label = pd.DataFrame(results_label, columns= ['label'])
    subs_pd = pd.concat([subs_images_pandas, results_label], axis = 1)
    results_submitted = pd.merge(test, subs_pd , on = 'image_id')
    results_submitted.to_csv(filename,index = False)

preds_subs_final = np.argmax(pred_test, axis = 1)

results_to_be_submitted(test = testx,result_subs = preds_subs_final  ,path = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/Submissions/',
                        name_of_file_sub = "results_xception") ##55.6% accuracy



###############Fine Tuning#######################################################
#############Loading the model####################

json_file = open('/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/model_Xception_10epochs.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
# load weights into new model
model.load_weights("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/model_no_augmentation_all_train_Xception_10epochs.h5")
print("Loaded model from disk")

for i, layer in enumerate(model.layers):
    print i ,layer.trainable


for i, layer in enumerate(model.layers):
    print i ,layer

####Freezing the top 15 layers###############
for layer in model.layers[:120]:
   layer.trainable = False
for layer in model.layers[120:]:
   layer.trainable = True

#Training the model now######################
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9), metrics=['accuracy'])

def train_model(epochs, train_generator):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=final_data_img.shape[0] // BATCH_SIZE,
        epochs=epochs,verbose = 1
)


BATCH_SIZE = 32

train_datagen = ImageDataGenerator()
train_datagen.fit(final_data_img)
train_generator = train_datagen.flow(final_data_img, final_labels_keras, batch_size=BATCH_SIZE)

start = time.time()
hist = train_model(10, train_generator)
end = time.time()

model_json = model.to_json()
with open("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/model_Xception_10epochs.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/model_no_augmentation_all_train_Xception_10epochs.h5")
print("Saved model to disk")


start = time.time()
pred_test = model.predict(subs_img, verbose= 1)
end = time.time()

joblib.dump(pred_test, '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/pred_test_xception')

def results_to_be_submitted(test,result_subs , path,name_of_file_sub):
    filename = path+name_of_file_sub+'.csv'
    image_names_subs = [re.search(r'(.*).png',text).group(1) for text in subs_images_files]
    subs_images_pandas = pd.DataFrame(image_names_subs, columns= ['image_id'])

    results_label = [dict_inverse_labels[item] for item in result_subs]
    results_label = pd.DataFrame(results_label, columns= ['label'])
    subs_pd = pd.concat([subs_images_pandas, results_label], axis = 1)
    results_submitted = pd.merge(test, subs_pd , on = 'image_id')
    results_submitted.to_csv(filename,index = False)

preds_subs_final = np.argmax(pred_test, axis = 1)

results_to_be_submitted(test = testx,result_subs = preds_subs_final  ,path = '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/Submissions/',
                        name_of_file_sub = "fine_tuned_results_xception") ##55.6% accuracy

joblib.dump(pred_test, '/home/ubuntu/linux/Work/Deep_Learning/Product_Classification/Data/a0409a00-8-dataset_dp/Dumps_for_Models/Fine_Tuned/pred_test_xception')