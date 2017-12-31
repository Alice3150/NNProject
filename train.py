import os  
import numpy as np  
import tensorflow as tf  
import input_data  
import model  
    
N_CLASSES = 25    
IMG_W = 100     
IMG_H = 100  
BATCH_SIZE = 40  
CAPACITY = 200  
MAX_STEP = 1000   
learning_rate = 0.0001   
  
#get batch  
train_dir = '/home/alice/Documents/NNProject/train_reshaped/'   #input path  
logs_train_dir = '/home/alice/Documents/NNProject/'    #logs path  
#logs_test_dir =  'E:/Re_train/image_data/test'          
 
#training set && validation set
train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)
  
train_batch,train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)  
 
val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)   
  
#training operation
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)  
train_loss = model.losses(train_logits, train_label_batch)          
train_op = model.trainning(train_loss, learning_rate)  
train_acc = model.evaluation(train_logits, train_label_batch)  
  
#validation operation
test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)  
test_loss = model.losses(test_logits, val_label_batch)          
test_acc = model.evaluation(test_logits, val_label_batch)  
  
#log  
summary_op = tf.summary.merge_all()   
 
sess = tf.InteractiveSession()    
 
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)   
#val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)   

saver = tf.train.Saver(var_list=tf.trainable_variables())  
#initialize 
sess.run(tf.global_variables_initializer())    
  
coord = tf.train.Coordinator()  
threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
  
#trianing  
try:    
    for step in np.arange(MAX_STEP):  
        if coord.should_stop():  
            break    
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])     
        if step % 10  == 0:  
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))  
            summary_str = sess.run(summary_op)  
            train_writer.add_summary(summary_str, step)    
        if (step + 1) == MAX_STEP:  
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
            saver.save(sess, checkpoint_path, global_step=step)  
         
except tf.errors.OutOfRangeError:  
    print('Done training -- epoch limit reached')  
  
finally:  
    coord.request_stop() 
