#from PIL import Image  
import numpy as np  
import tensorflow as tf  
#import matplotlib.pyplot as plt  
import model  
#from input_data import get_files
  
  
#--------------------------------------------------------------------    
def evaluate_one_image(image_array):  
    with tf.Graph().as_default():  
       BATCH_SIZE = 1  
       N_CLASSES = 25  
  
       image = tf.cast(image_array, tf.float32) 
       image = tf.image.per_image_standardization(image)  
       image = tf.reshape(image, [1, 100, 100, 3])  
  
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)  
  
       logit = tf.nn.softmax(logit)  
  
       x = tf.placeholder(tf.float32, shape=[100, 100, 3])  
    
       logs_train_dir = './'  
  
       saver = tf.train.Saver()  
  
       with tf.Session() as sess:  
  
           #print("Reading checkpoints...")  
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
           if ckpt and ckpt.model_checkpoint_path:  
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
               saver.restore(sess, ckpt.model_checkpoint_path)  
               #print('Loading success, global_step is %s' % global_step)  
           else:  
               print('No checkpoint file found')  
  
           prediction = sess.run(logit, feed_dict={x: image_array})  
           max_index = np.argmax(prediction)
           #print(max_index)  
           return max_index
           # if max_index==0:  
           #     print('This is a american bulldog with possibility %.6f' %prediction[:, 0])  
           # elif max_index==1:  
           #     print('This is a american pit bull terrier with possibility %.6f' %prediction[:, 1])  
           # elif max_index==2:  
           #     print('This is a basset_hound with possibility %.6f' %prediction[:, 2])  
           # elif max_index==3:  
           #     print('This is a beagle with possibility %.6f' %prediction[:, 3])
           # elif max_index==4:
           #     print('This is a boxer with possibility %.6f' %prediction[:, 4])
           # elif max_index==5:
           #     print('This is a chihuahua with possibility %.6f' %prediction[:, 5])
           # elif max_index==6:
           #     print('This is a english cocker spainel with possibility %.6f' %prediction[:, 6])
           # elif max_index==7:
           #     print('This is a english setter with possibility %.6f' %prediction[:, 7])
           # elif max_index==8:
           #     print('This is a german shorthaired with possibility %.6f' %prediction[:, 8])
           # elif max_index==9:
           #     print('This is a great pyrenees with possibility %.6f' %prediction[:, 9])
           # elif max_index==10:
           #     print('This is a havanese with possibility %.6f' %prediction[:, 10])
           # elif max_index==11:
           #     print('This is a japanese chin with possibility %.6f' %prediction[:, 11])
           # elif max_index==12:
           #     print('This is a keeshond with possibility %.6f' %prediction[:, 12])
           # elif max_index==13:
           #     print('This is a leonberger with possibility %.6f' %prediction[:, 13])
           # elif max_index==14:
           #     print('This is a miniature pinscher with possibility %.6f' %prediction[:, 14])
           # elif max_index==15:
           #     print('This is a newfoundland with possibility %.6f' %prediction[:, 15])
           # elif max_index==16:
           #     print('This is a pomeranian with possibility %.6f' %prediction[:, 16])
           # elif max_index==17:
           #     print('This is a pug with possibility %.6f' %prediction[:, 17])
           # elif max_index==18:
           #     print('This is a saint bernard with possibility %.6f' %prediction[:, 18])
           # elif max_index==19:
           #     print('This is a samoyed with possibility %.6f' %prediction[:, 19])
           # elif max_index==20:
           #     print('This is a scottish terrier with possibility %.6f' %prediction[:, 20])
           # elif max_index==21:
           #     print('This is a shiba inu with possibility %.6f' %prediction[:, 21])
           # elif max_index==22:
           #     print('This is a staffordshire bull terrier with possibility %.6f' %prediction[:, 22])
           # elif max_index==23:
           #     print('This is a wheaten terrier with possibility %.6f' %prediction[:, 23])
           # else:
           #     print('This is a yorkshire terrier with possibility %.6f' %prediction[:, 24])
  
#------------------------------------------------------------------------  
#=======================================================================
               
if __name__ == '__main__':  
    
    test_data = "../testing.npy"
    tmp = np.load(test_data).item()
    reshaped = tmp["reshaped"]
    logits = []
    for img in reshaped: 
        img_reshaped = np.reshape(img, [100, 100, 3])
        img_ = img_reshaped.astype(np.uint8)
        prediction = evaluate_one_image(img_)
        logits.append(prediction)
    np.savetxt('labels.txt', logits, delimiter='\n', fmt='%d')
