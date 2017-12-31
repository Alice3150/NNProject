import tensorflow as tf
import numpy as np
import os
import math
# import matplotlib.pyplot as plt

train_dir = '/home/alice/Documents/NNProject/train_reshaped/'#your data directory
def get_files(file_dir, ratio):
    american_bulldog = []
    label_american_bulldog = [] #0
    american_pit = []
    label_american_pit = [] #1
    basset = []
    label_basset = [] #2
    beagle = []
    label_beagle = [] #3
    boxer = []
    label_boxer = [] #4
    chihuahua = []
    label_chihuahua = [] #5
    english_cocker = []
    label_english_cocker = [] #6
    english_setter = []
    label_english_setter = [] #7
    german = []
    label_german = [] #8
    great = []
    label_great = [] #9
    havanese = []
    label_havanese = [] #10
    japanese = []
    label_japanese = [] #11
    keeshond = []
    label_keeshond = [] #12
    leonberger = []
    label_leonberger = [] #13
    miniature = []
    label_miniature = [] #14
    newfoundland = []
    label_newfoundland = [] #15
    pomeranian = []
    label_pomeranian = [] #16
    pug = []
    label_pug = [] #17
    saint = []
    label_saint = [] #18
    samoyed = []
    label_samoyed = [] #19
    scottish = []
    label_scottish = [] #20
    shiba = []
    label_shiba = [] #21
    staffordshire = []
    label_staffordshire = [] #22
    wheaten = []
    label_wheaten = [] #23
    yorkshire = []
    label_yorkshire = [] #24

    for file in os.listdir(file_dir):
        name = file.split('_')
        if name[0]=="american":
            if name[1] == "bulldog":
                american_bulldog.append(file_dir + file)
                label_american_bulldog.append(0)
            elif name[1] == "pit":
                american_pit.append(file_dir + file)
                label_american_pit.append(1)
        elif name[0]=="basset":
            basset.append(file_dir + file)
            label_basset.append(2)
        elif name[0]=="beagle":
        	beagle.append(file_dir + file)
        	label_beagle.append(3)
        elif name[0]=="boxer":
        	boxer.append(file_dir + file)
        	label_boxer.append(4)
        elif name[0]=="chihuahua":
        	chihuahua.append(file_dir + file)
        	label_chihuahua.append(5)
        if name[0]=="english":
            if name[1]=="cocker":
                english_cocker.append(file_dir + file)
                label_english_cocker.append(6)
            elif name[1]=="setter":
                english_setter.append(file_dir + file)
                label_english_setter.append(7)
        elif name[0]=="german":
        	german.append(file_dir + file)
        	label_german.append(8)
        elif name[0]=="great":
        	great.append(file_dir + file)
        	label_great.append(9)
        elif name[0]=="havanese":
        	havanese.append(file_dir + file)
        	label_havanese.append(10)
        elif name[0]=="japanese":
        	japanese.append(file_dir + file)
        	label_japanese.append(11)
        elif name[0]=="keeshond":
        	keeshond.append(file_dir + file)
        	label_keeshond.append(12)
        elif name[0]=="leonberger":
        	leonberger.append(file_dir + file)
        	label_leonberger.append(13)
        elif name[0]=="miniature":
        	miniature.append(file_dir + file)
        	label_miniature.append(14)
        elif name[0]=="newfoundland":
        	newfoundland.append(file_dir + file)
        	label_newfoundland.append(15)
        elif name[0]=="pomeranian":
        	pomeranian.append(file_dir + file)
        	label_pomeranian.append(16)
        elif name[0]=="pug":
        	pug.append(file_dir + file)
        	label_pug.append(17)
        elif name[0]=="saint":
        	saint.append(file_dir + file)
        	label_saint.append(18)
        elif name[0]=="samoyed":
        	samoyed.append(file_dir + file)
        	label_samoyed.append(19)
        elif name[0]=="scottish":
        	scottish.append(file_dir + file)
        	label_scottish.append(20)
        elif name[0]=="shiba":
        	shiba.append(file_dir + file)
        	label_shiba.append(21)
        elif name[0]=="staffordshire":
        	staffordshire.append(file_dir + file)
        	label_staffordshire.append(22)
        elif name[0]=="wheaten":
        	wheaten.append(file_dir + file)
        	label_wheaten.append(23)
        elif name[0]=="yorkshire":
        	yorkshire.append(file_dir + file)
        	label_yorkshire.append(24)

    #print('There are %d iris\nThere are %d contact' %(len(iris), len(contact)))

    image_list = np.hstack((american_bulldog, american_pit, basset, beagle, boxer, chihuahua, english_cocker, english_setter, german, great, havanese, japanese, keeshond, leonberger, miniature, newfoundland, pomeranian, pug, saint, samoyed, scottish, shiba, staffordshire, wheaten, yorkshire))
    label_list = np.hstack((label_american_bulldog, label_american_pit, label_basset, label_beagle, label_boxer, label_chihuahua, label_english_cocker, label_english_setter, label_german, label_great, label_havanese, label_japanese, label_keeshond, label_leonberger, label_miniature, label_newfoundland, label_pomeranian, label_pug, label_saint, label_samoyed, label_scottish, label_shiba, label_staffordshire, label_wheaten, label_yorkshire))

    #shuffle
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    n_sample = len(all_label_list)  
    n_val = int(math.ceil(n_sample*ratio))    
    n_train = n_sample - n_val     
  
    tra_images = all_image_list[0:n_train]  
    tra_labels = all_label_list[0:n_train]  
    tra_labels = [int(float(i)) for i in tra_labels]  
    val_images = all_image_list[n_train:-1]  
    val_labels = all_label_list[n_train:-1]  
    val_labels = [int(float(i)) for i in val_labels]  
  
    return tra_images, tra_labels, val_images, val_labels 

def get_batch(image, label, image_W, image_H, batch_size, capacity):  
  
    image = tf.cast(image, tf.string)  
    label = tf.cast(label, tf.int32)  
  
    # make an input queue  
    input_queue = tf.train.slice_input_producer([image, label])  
  
    label = input_queue[1]  
    image_contents = tf.read_file(input_queue[0]) #read img from a queue    
        
    image = tf.image.decode_jpeg(image_contents, channels=3)   
        
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  
    image = tf.image.per_image_standardization(image)  
  
    image_batch, label_batch = tf.train.batch([image, label],  
                                                batch_size= batch_size,  
                                                num_threads= 32,   
                                                capacity = capacity)  
  
    label_batch = tf.reshape(label_batch, [batch_size])  
    image_batch = tf.cast(image_batch, tf.float32)  
    return image_batch, label_batch
