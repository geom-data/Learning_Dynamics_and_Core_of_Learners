# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 08:24:48 2023

@author: Siu
"""

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras import layers
from keras.layers import Dense, Concatenate, BatchNormalization, Dropout, Conv2D, Flatten, LeakyReLU, Activation, add
from keras.layers import AveragePooling2D, Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import Model, load_model, clone_model
from keras.losses import MSE
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import math
import numpy as np
import pickle
import os
import importlib
import pandas as pd
from IPython.display import display

from resnet import ResNet

#Create logifold by: Logifold(x_tr,y_tr,x_v,y_v,target,path)
#target is the number of classes.
#Then use load() to load pretrained models saved in path as charts.
#Then take getFuzDoms() to compute the fuzzy domains for each chart.
#Can then predict or evaluate.
#Then can train/turn specialists and blowup to improve the coverage of atlas.
#If use datagen or dataset, assume data is placed at path/data/tr and path/data/val
#If ds_tr_info is not provided, then use default Keras method
#Alternatively, can provide x_tr as the whole dataset.  y_tr=None for ds or datagen.
class Logifold:
    def __init__(self, target : int, dataType : str ="numpy",
                name : str=None, # society name
                x_tr=None, y_tr=None, x_v=None, y_v=None,
                path : str =None, load_val_path : bool =False, load_tr_path : bool =False,
                ds_tr_info=None, ds_val_info=None, #this is for dataType "datagen" or "ds" only
                image_size=None, color_range:float=None, batch_size:int=32,#this is for using default keras datagen or ds method only
                input_shape=None,
                storyFile : str =None, new_story=True):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_v = x_v
        self.y_v = y_v
        self.dataType = dataType
        self.ds_tr_info = ds_tr_info
        self.ds_val_info = ds_val_info
        self.path = path #"path_of_folder/" for saving and loading
        self.target = target #|target set|.
        self.name = name
        self.valDom = None
        self.storyFile = storyFile
        self.committees = {}
        self.raw_predictions = None
        #This is for later versions.
        #ex. "animals":{2,3,5,7}, "automobiles":{1,4,6}
        #Then we can run self.predict for a small part of given test data
        #to check whether it matches with any contexts stored here.
        #If there is match in high acc, we try to set to be the corresponding target.
        self.contexts = {}
        
        #load data
        if dataType == "numpy":
            if x_v is None and load_val_path:
                with open(path+'y_v.npy', 'rb') as f:
                    self.y_v = np.load(f)
                if os.path.isfile(path+'x_v.npy'):
                    with open(path+'x_v.npy', 'rb') as f:
                        self.x_v = np.load(f)
                else:
                    i=0
                    self.x_v=[]
                    while os.path.isfile(path+'x_v%d.npy'%i):
                        with open(path+'x_v%d.npy'%i, 'rb') as f:
                            self.x_v.append(np.load(f))
                        i+=1
                    self.x_v = np.concatenate(self.x_v,axis=0)
            if x_tr is None and load_tr_path:
                with open(path+'y_tr.npy', 'rb') as f:
                    self.y_tr = np.load(f)            
                if os.path.isfile(path+'x_tr.npy'):
                    with open(path+'x_tr.npy', 'rb') as f:
                        self.x_tr = np.load(f)
                else:
                    i=0
                    self.x_tr=[]
                    while os.path.isfile(path+'x_tr%d.npy'%i):
                        with open(path+'x_tr%d.npy'%i, 'rb') as f:
                            self.x_tr.append(np.load(f))
                        i+=1       
                    self.x_tr = np.concatenate(self.x_tr,axis=0)                    
        elif dataType == "ds":    
            if x_tr is None and ds_tr_info is not None:
                print("Constructing training dataset from path/data/tr/...")
                if isinstance(ds_tr_info,dict):
                    module = ds_tr_info['module']
                    method = ds_tr_info['method']
                    optarg = ds_tr_info['optarg']
                    m = importlib.import_module(module)
                    f = getattr(m, method)
                    self.x_tr = f(self.path + 'data/tr/',**optarg)
                else: #Assume provided ds_tr_info is a method to load data from path
                    self.x_tr = ds_tr_info(self.path + 'data/tr/')
            #Default method using Keras:
            else:
                print("Using Keras method image_dataset_from_directory to construct training dataset from path/data/tr/...")
                self.ds_tr_info = {}
                self.ds_tr_info['module'] = 'tensorflow.keras.utils'
                self.ds_tr_info['method'] = 'image_dataset_from_directory'
                self.ds_tr_info['optarg'] = {'labels':'inferred', 
                                             'label_mode':'categorical',
                                             'batch_size':batch_size}
                    
                if image_size is not None:
                    self.ds_tr_info['optarg']['image_size'] = image_size
                m = importlib.import_module('tensorflow.keras.utils')
                f = getattr(m, 'image_dataset_from_directory')
                optarg = self.ds_tr_info['optarg']
                self.x_tr = f(self.path + 'data/tr/',**optarg)                    
            if x_v is None and ds_val_info is not None:
                print("Constructing validation dataset from path/data/val/...")
                if isinstance(ds_val_info,dict):
                    module = ds_val_info['module']
                    method = ds_val_info['method']
                    optarg = ds_val_info['optarg']
                    m = importlib.import_module(module)
                    f = getattr(m, method)
                    self.x_v = f(self.path + 'data/val/',**optarg)
                else:
                    self.x_v = ds_val_info(self.path + 'data/val/')
            #Default method using Keras:
            else:
                print("Using Keras method image_dataset_from_directory to construct validation dataset from path/data/val/...")
                self.ds_val_info = {}
                self.ds_val_info['module'] = 'tensorflow.keras.utils'
                self.ds_val_info['method'] = 'image_dataset_from_directory'
                self.ds_val_info['optarg'] = {'labels':'inferred',
                                              'label_mode':'categorical',
                                              'batch_size':batch_size}
                if image_size is not None:
                    self.ds_val_info['optarg']['image_size'] = image_size                  
                m = importlib.import_module('tensorflow.keras.utils')
                f = getattr(m, 'image_dataset_from_directory')
                optarg = self.ds_tr_info['optarg']
                self.x_v = f(self.path + 'data/val/',**optarg)                     
        elif dataType == "datagen":
            if x_tr is None and ds_tr_info is not None:
                if isinstance(ds_tr_info,dict):
                    print("Constructing training datagen from path/data/tr/...")
                    #ex. 
                    #'module':'tensorflow.keras.preprocessing.image',
                    #'method':'ImageDataGenerator'
                    #"optarg_datagen":{"rescale":1./255}
                    #'flow_method':'flow_from_directory'
                    #'optarg_flow' = {'class_mode':'categorical', 
                    #                 "target_size":(32,32), 'batch_size':64}
                    module = ds_tr_info['module']
                    method = ds_tr_info['method']
                    optarg_datagen = ds_tr_info['optarg_datagen']
                    flow_method = ds_tr_info['flow_method']
                    optarg_flow = ds_tr_info['optarg_flow']
                    m = importlib.import_module(module)
                    datagen = getattr(m, method)(self.path + 'data/tr/',**optarg_datagen)
                    flow = getattr(datagen, flow_method)
                    self.x_tr = flow(self.path + 'data/tr/', **optarg_flow)
                else: #Assume provided ds_tr_info is a method to genenerate data flow from path
                    self.x_tr = ds_tr_info(self.path + 'data/tr/')
            #Default method using Keras:
            else:
                print("Using Keras method image_dataset_from_directory to construct training datagen from path/data/tr/...")
                self.ds_tr_info = {}
                self.ds_tr_info['module'] = 'tensorflow.keras.preprocessing.image'
                self.ds_tr_info['method'] = 'ImageDataGenerator'
                self.ds_tr_info['optarg_datagen'] = {}
                if color_range is not None:
                    self.ds_tr_info['optarg_datagen']['rescale'] = 1./color_range
                self.ds_tr_info['flow_method'] = 'flow_from_directory'    
                self.ds_tr_info['optarg_flow'] = {'class_mode':'categorical','batch_size':batch_size}                 
                if image_size is not None:
                    self.ds_tr_info['optarg_flow']['target_size'] = image_size                    
                m = importlib.import_module('tensorflow.keras.preprocessing.image')
                optarg_datagen = self.ds_tr_info['optarg_datagen']
                datagen = getattr(m, 'ImageDataGenerator')(self.path + 'data/tr/',**optarg_datagen)
                flow = getattr(datagen, 'flow_from_directory')
                optarg_flow = self.ds_tr_info['optarg_flow']
                self.x_tr = flow(self.path + 'data/tr/', **optarg_flow)                        
            if x_v is None and ds_val_info is not None:
                if isinstance(ds_val_info,dict):
                    print("Constructing validation datagen from path/data/val/...")
                    #ex. 
                    #'module':'tensorflow.keras.preprocessing.image',
                    #'method':'ImageDataGenerator'
                    #'flow_method':'flow_from_directory'
                    #'optarg_flow' = {'class_mode':'categorical', 'batch_size':64}
                    module = ds_val_info['module']
                    method = ds_val_info['method']
                    optarg_datagen = ds_val_info['optarg_datagen']
                    flow_method = ds_val_info['flow_method']
                    optarg_flow = ds_val_info['optarg_flow']
                    m = importlib.import_module(module)
                    datagen = getattr(m, method)(self.path + 'data/val/',**optarg_datagen)
                    flow = getattr(datagen, flow_method)
                    self.x_v = flow(self.path + 'data/val/', **optarg_flow)
                else: #Assume provided ds_tr_info is a method to genenerate data flow from path
                    self.ds_val_info = {}
                    self.x_v = ds_val_info(self.path + 'data/val/')
            #Default method using Keras:
            else:
                print("Using Keras method image_dataset_from_directory to construct validation datagen from path/data/val/...")
                self.ds_val_info = {}
                self.ds_val_info['module'] = 'tensorflow.keras.preprocessing.image'
                self.ds_val_info['method'] = 'ImageDataGenerator'
                self.ds_val_info['optarg_datagen'] = {}
                if color_range is not None:
                    self.ds_val_info['optarg_datagen']['rescale'] = 1./color_range
                self.ds_val_info['flow_method'] = 'flow_from_directory'    
                self.ds_val_info['optarg_flow'] = {'class_mode':'categorical','batch_size':batch_size}        
                if image_size is not None:
                    self.ds_val_info['optarg_flow']['target_size'] = image_size                    
                m = importlib.import_module('tensorflow.keras.preprocessing.image')
                optarg_datagen = self.ds_val_info['optarg_datagen']
                datagen = getattr(m, 'ImageDataGenerator')(self.path + 'data/val/',**optarg_datagen)
                flow = getattr(datagen, 'flow_from_directory')
                optarg_flow = self.ds_val_info['optarg_flow']
                self.x_v = flow(self.path + 'data/val/', **optarg_flow)            
        else:
            print("Data types other than numpy, dataset and datagen are not supported.  Data is not loaded.")
        
        #Detect input_shape
        if dataType == "numpy":
            if self.x_tr is not None:
                self.input_shape = self.x_tr.shape[1:]
            elif self.x_v is not None:
                self.input_shape = self.x_v.shape[1:]
        elif dataType == "ds" or dataType == "datagen":
            if self.x_tr is not None:
                self.input_shape = next(iter(self.x_tr))[0].shape[1:]
            elif self.x_v is not None:
                self.input_shape = next(iter(self.x_v))[0].shape[1:]
        else:
            self.input_shape=input_shape        
        
        self.charts = {}
        #This is a dictionary of charts. 
        #Each chart consists of a dictionary with two keys: fuzDom and target.
        #The dictionary keys one-one correspond to names of models in path.
        #Assume keys are arrays of integers.
        #So each chart is essentially the graph of a function with fuzzy domain and target.
        #We also add another key called "active" to the dictionary.
        #It is set to True by default.
        #We can set it to False to deactivate a chart. 
        #We also add "filetype" key to save the file type of the model file (ex h5).
        #We also add "description" key to allow adding descriptions of models.
        #This will be useful when there are many models.
        
        if path is not None and not os.path.isdir(path):
            os.mkdir(path)
        
        if new_story and storyFile is not None:    
            story = open(self.path+storyFile+".txt","w")
            L = ["---Model Society Started---\n"]
            if name is not None:
                L.append("Our society is named %s.\n"%name)
            if dataType == "numpy":
                if self.x_tr is not None:
                    L.append("Training sample size is %d.\n"%len(self.x_tr))
                if self.x_v is not None:
                    L.append("Validation sample size is %d.\n"%len(self.x_v))
            elif dataType == "ds" or dataType == "datagen":
                if self.x_tr is not None:
                    L.append("Training sample has %d batches with batch size %d.\n"%(len(self.x_tr),batch_size))
            story.writelines(L)
            story.close()

            
    #archi=[CNN_array, dense_array]. 
    #CNN_array=[[[thickness,kernelSize],...,dropout_rate],...]
    #where thickness*8 will be taken as the actual thickness of CNN layer
    #dense_array=[[size,dropout_rate],...]
    #Note that size=1 already means 8 units.
    #CNN_array can be [].            
    def genCNN(self, key, filetype='h5', target=None, 
               archi=[[[[[2,3],[4,3]],0.25],[[[4,3],[8,3]],0.25]],[[32,0.5],[1,0.]]],
               epochs : int =50,lr : float =1e-3,batch_size : int =32,
               iniTh : float =0.,step : float =0.5,minValPercentage : float =0.01,datagen=None,
               train : bool =True,active : bool =True, save_each : bool =False, verbose:int=1,
               saveAboveAcc:float=None,saveBests:int=None, lr_schedule=None,
               description : str =None, replace_description:bool=False,
               spec=None,autosave : bool =True):        
        assert self.input_shape is not None
        input_shape = self.input_shape
        if target is None:
            n_classes = self.target
            target = [[i] for i in range(self.target)]
        else:
            n_classes = len(target)
        inputs = keras.Input(shape=input_shape)
        y = inputs
        for convBlock in archi[0]:
            for conv in convBlock[0]:
                y = Conv2D(8*conv[0], kernel_size=(conv[1], conv[1]),padding='same')(y)
                y = LeakyReLU(alpha=0.1)(y)
            y = layers.MaxPooling2D(pool_size=(2, 2))(y) 
            y = BatchNormalization()(y)
            y = Dropout(convBlock[1])(y)
        y = Flatten()(y)
        for dens in archi[1]:
            y = Dense(8*dens[0])(y)
            y = LeakyReLU(alpha=0.1)(y)
            y = Dropout(dens[1])(y)       
        y = Dense(n_classes)(y)
        outputs = layers.Softmax()(y)
        model = Model(inputs=inputs, outputs=outputs)
        
        filepath = self.modelPath(key,filetype)
        model.save(filepath)
        self.charts[key] = {}
        self.charts[key]['fuzDom'] = None
        self.charts[key]['target'] = target
        self.charts[key]['active'] = active
        self.charts[key]['filetype'] = filetype
        self.charts[key]['spec for loading model'] = None

        text = ""
        if not replace_description:
            text += "Model made by genCNN method.\n"
        if description is not None:
            text += description + "\n"       
        self.charts[key]['description'] = text
        
        storyFile = self.storyFile
        if storyFile is not None:    
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---New CNN model built---\n")
                story.write(f"Its key is {key}.\n")   
                
        #Train if there is training data
        if train and self.x_tr is not None and self.y_tr is not None:            
            train_summary=self.train(key,epochs=epochs,lr=lr,
                               batch_size=batch_size,datagen=datagen,
                               save_each=save_each, saveAboveAcc=saveAboveAcc, saveBests=saveBests,
                               lr_schedule=lr_schedule, autosave=autosave, verbose=verbose)
            if storyFile is not None:    
                with open(self.path+storyFile+".txt", "a") as story:
                    story.write(train_summary + '\n')
        else:   
            self.getFuzDoms(iniTh=iniTh, step=step, 
                       minValPercentage=minValPercentage, keys=[key], autosave=autosave)
        return

    def resnet(self, key, version=1, n=3, 
               train : bool =True, epochs : int =200, datagen='default', lr_schedule='default',
               filetype='h5', target=None, 
               lr : float =1e-3,batch_size : int =32,
               iniTh : float =0.,step : float =0.5,minValPercentage : float =0.01,
               active : bool =True, save_each : bool =False, verbose:int=1,
               saveAboveAcc:float=None, saveBests:int=None,
               description : str =None, replace_description:bool=False,
               autosave : bool =True):        
        assert self.input_shape is not None
        input_shape = self.input_shape
        if target is None:
            n_classes = self.target
            target = [[i] for i in range(self.target)]
        else:
            n_classes = len(target)

        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2        
            
        if version == 2:
            model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=n_classes)
        else:
            model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=n_classes)            

        filepath = self.modelPath(key,filetype)
        model.save(filepath)
        self.charts[key] = {}
        self.charts[key]['fuzDom'] = None
        self.charts[key]['target'] = target
        self.charts[key]['active'] = active
        self.charts[key]['filetype'] = filetype
        self.charts[key]['spec for loading model'] = None

        text = ""
        if not replace_description:
            text += "Model made by genCNN method.\n"         
        if description is not None:
            text += description + "\n"       
        self.charts[key]['description'] = text        
        
        storyFile = self.storyFile
        if storyFile is not None:    
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---New ResNet built---\n")
                story.write(f"Its key is {key}.\n")   
                
        if train and self.x_tr is not None:
            if datagen == 'default' and self.dataType=='numpy':
                datagen = ImageDataGenerator(
                    # set input mean to 0 over the dataset
                    featurewise_center=False,
                    # set each sample mean to 0
                    samplewise_center=False,
                    # divide inputs by std of dataset
                    featurewise_std_normalization=False,
                    # divide each input by its std
                    samplewise_std_normalization=False,
                    # apply ZCA whitening
                    zca_whitening=False,
                    # randomly rotate images in the range (deg 0 to 180)
                    rotation_range=0,
                    # randomly shift images horizontally
                    width_shift_range=0.1,
                    # randomly shift images vertically
                    height_shift_range=0.1,
                    # randomly flip images
                    horizontal_flip=True,
                    # randomly flip images
                    vertical_flip=False)                
            if lr_schedule == 'default':
                def lr_schedule(epoch):
                    """Learning Rate Schedule
                
                    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
                    Called automatically every epoch as part of callbacks during training.
                
                    # Arguments
                        epoch (int): The number of epochs
                
                    # Returns
                        lr (float32): learning rate
                    """
                    lr = 1e-3
                    if epoch > 180:
                        lr *= 0.5e-3
                    elif epoch > 160:
                        lr *= 1e-3
                    elif epoch > 120:
                        lr *= 1e-2
                    elif epoch > 80:
                        lr *= 1e-1
                    print('Learning rate: ', lr)
                    return lr                
                
            train_summary=self.train(key,epochs=epochs,lr=lr,
                               batch_size=batch_size,datagen=datagen,
                               save_each=save_each,saveAboveAcc=saveAboveAcc, saveBests=saveBests,
                               verbose=verbose,
                               lr_schedule=lr_schedule,autosave=autosave)
            if storyFile is not None:    
                with open(self.path+storyFile+".txt", "a") as story:
                    story.write(train_summary + '\n')
        else:   
            self.getFuzDoms(iniTh=iniTh, step=step, 
                       minValPercentage=minValPercentage, keys=[key], autosave=autosave)
        return
    
    def addKerasPretrainedModel(self, module, method, optarg=None, active=True, description=None):
        assert isinstance(module,str)
        assert isinstance(method,str)
        # Assign a new key
        n,m = len(module), len(method)
        if optarg is not None:
            assert isinstance(optarg,dict)
            additional_key = int()
            for _ in optarg.keys():
                additional_key += len(_)
            key = (n,m,additional_key)
        else:
            key = (n,m)
        while key in self.keys():
            n +=1
            if optarg is not None:
                key = (n,m,additional_key)
            else:
                key = (n,m)
        # Assing value to the newly generated key
        self.charts[key] = {}
        self.charts[key]['fuzDom'] = None
        self.charts[key]['target'] = [[j] for j in range(self.target)]
        self.charts[key]['active'] = active
        self.charts[key]['filetype'] = None
        self.charts[key]['description'] = description
        self.charts[key]['spec for loading model'] = {'name':'Keras-pretrained',
                                                      'module':module,
                                                      'method':method,
                                                      'optarg':optarg}
        storyFile = self.storyFile
        if storyFile is not None:    
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---Welcome new member---\n")
                story.write(f"Keras Pretrained model {module}(method = {method}) is added with key {key}.\n")
        return
    
    def isFull(self, target):
        l=0
        if isinstance(target,tuple):
            target = self.charts[target]['target']
        for t in target:
            l+=len(t)
        return l==self.target
    
    def isFine(self, target):
        initial = True
        if isinstance(target,tuple):
            target = self.charts[target]['target']        
        for t in target:
            initial = len(t)==1
            if not initial:
                break
        return initial
    
    def isComplete(self, target):
        if isinstance(target,tuple):
            target = self.charts[target]['target']   
        return self.isFull(target) and self.isFine(target)

    def orderByAcc(self, keys=None, active=False, 
                   cl=None, flatTarget=None, fine=True):
        if keys is None:
            keys = self.keys()
        if active:
            keys = [k for k in keys if self.charts[k]['active']]
        if flatTarget is None:
            flatTarget = tuple(range(self.target))
        keys = self.keysInFlatTarget(flatTarget, keys=keys, active=active, fine=fine)

        if cl is None:
            accs = np.array([self.charts[k]['fuzDom'][-float('inf')][0][0] 
                             for k in keys])
        else:
            assert cl in range(len(flatTarget))
            #Only consider fine models to order by class accuracy
            #Also, assume ascending order of indices for each t in target
            #ex. [1,2] is valid, [2,1] is not
            keys = [k for k in keys if len(self.charts[k]['target']) == len(flatTarget)]
            accs = np.array([self.charts[k]['fuzDom'][-float('inf')][1][cl] 
                             for k in keys])
        order = np.flip(np.argsort(accs))  #descending order
        return [keys[i] for i in order]
    
    def keysAboveAcc(self,acc,keys=None,active=False):
        if keys is None:
            keys = self.keys()
        if active:
            keys = [k for k in keys if self.charts[k]['active']]
        return [key for key in keys 
                if self.charts[key]['fuzDom'][-float('inf')][0][0] > acc]
    
    def acc(self, key):
        return self.charts[key]['fuzDom'][-float('inf')][0][0]
    
    def accs(self, keys=None, active=False):
        if keys is None:
            keys = self.keys()
        if active:
            return np.array([self.charts[key]['fuzDom'][-float('inf')][0][0] 
                    for key in keys if self.charts[key]['active']])
        else:
            return np.array([self.charts[key]['fuzDom'][-float('inf')][0][0] 
                    for key in keys])

    def accStat(self,keys=None, active=False):
        accs = self.accs(keys=keys,active=active)
        accStat={}
        accStat['med'] = np.median(accs)
        accStat['mean'] = np.mean(accs)
        accStat['max'] = np.max(accs)
        accStat['min'] = np.min(accs)
        accStat['75% percent'] = np.percentile(accs,75)
        accStat['topMean'] = np.mean(accs[accs>=accStat['75% percent']])
        return accStat

    def activate(self, keys=None, acc_above=None):      
        if keys is None:
            keys = self.keys()
        if not isinstance(keys,list):
            keys = [keys]
        if acc_above is None:
            for key in keys:
                self.charts[key]['active'] = True
        else:
            for key in keys:
                if self.charts[key]['fuzDom'][-float('inf')][0][0] > acc_above:
                    self.charts[key]['active'] = True
        return
    
    def deactivate(self, keys=None, acc_below=None):
        assert keys is not None or acc_below is not None
        if keys is None:
            keys = self.keys()
        if not isinstance(keys,list):
            keys = [keys]
        if acc_below is None:
            for key in keys:
                self.charts[key]['active'] = False
        else:
            for key in keys:
                if self.charts[key]['fuzDom'][-float('inf')][0][0] < acc_below:
                    self.charts[key]['active'] = False            
        return    
    
    def add(self,model,key : tuple,filetype : str,target=None,
           iniTh : float =0., step : float =0.5, minValPercentage : float =0.01, active=True,
           description : str=None, verbose : int = 0, fuzDom : dict = None):
        model.save(self.modelPath(key,filetype))
        if target is None:
            target = [[i] for i in range(self.target)]
        self.charts[key] = {}
        self.charts[key]['target'] = target
        self.charts[key]['active'] = active
        self.charts[key]['filetype'] = filetype
        self.charts[key]['description'] = description
        self.charts[key]['spec for loading model'] = None
        if fuzDom is None:
            self.getFuzDoms([key],iniTh=iniTh, step=step, 
                       minValPercentage=minValPercentage, verbose = verbose)
        else:
            self.charts[key]['fuzDom'] = fuzDom
        storyFile = self.storyFile
        if storyFile is not None:    
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---Welcome new member---\n")
                story.write(f"model-{key} is added. Now our society has {len(self.charts.keys())} number of models.\n")
        return
                        
    def getModel(self,key):        
        if self.charts[key]['spec for loading model'] is not None:
            if self.charts[key]['spec for loading model']['name'] == 'scale_in':
                scale = self.charts[key]['spec for loading model'][0]
                softmax_k = softmax_scaled(scale)
                filetype = self.charts[key]['filetype']
                model = load_model(self.modelPath(key,filetype),custom_objects={'softmax_k':softmax_k})
            elif self.charts[key]['spec for loading model']['name'] == 'Keras-pretrained':
                module = self.charts[key]['spec for loading model']['module']
                method = self.charts[key]['spec for loading model']['method']
                optarg = self.charts[key]['spec for loading model']['optarg']
                m = importlib.import_module(module)
                f = getattr(m, method)
                model = f(**optarg)                
            else:
                print('Cannot regconize the specification of Chart %s'%(self.modelName(key)))
                return                
        else:
            filetype = self.charts[key]['filetype']
            model = load_model(self.modelPath(key,filetype))
        return model
    
    def modelPath(self,key,filetype):
        s = ''
        for i in key:
            s+='_%03d'%i
        return self.path+'model'+s+'.'+filetype
    
    def modelName(self,key):
        s = ''
        for i in key:
            s+='_%03d'%i
        return 'model' + s        
    
    def force_batch_model_predict(self, model, x, batch_size):
        x_split = np.array_split(x,math.ceil(len(x)/batch_size))
        predict = []
        for batch in x_split:
            predict.append(model.predict(x,verbose=0))
        predict = np.concatenate(predict,axis=-1)  
        return predict
    
    #save below just means to save to memory for logifold.
    #autosave means to save to file.
    #TODO verbosity level = 1 then show which model computes its fuzzy domain.
    def getFuzDoms(self, x=None, y=None, iniTh=0., step=0.5, 
                   minValPercentage=0.01, save=True, keys=None,batch_size=128,
                   force_batch=False, onlyForActive=False, update=True,
                   autosave=True,verbose : int = 0):
        #Get thresholds of model[k] for k in keys using validation data
        if keys is None:
            keys=self.charts.keys()
        if update:
            keys=[k for k in keys if 'fuzDom' not in self.charts[k] or self.charts[k]['fuzDom'] is None]
        
        if x is None or y is None:
            x = self.x_v
            y = self.y_v
        
        minValNo = len(y) * minValPercentage
        
        print("Computing fuzzy domains...")
        fuzDoms = {}
        for k in keys:
            if onlyForActive and not self.charts[k]['active']:
                continue
            
            model = self.getModel(k)
            
            target=self.charts[k]['target']
            
            x_k, y_k = self.newLabelForTarget(x, y, target, digit=True)
            if force_batch == True:
                predict = self.force_batch_model_predict(model, x_k, batch_size)      
            else:
                predict=model.predict(x_k,batch_size=batch_size,verbose=verbose)
            assert len(predict.shape)==2
            ans = np.argmax(predict,axis=-1)
            certainty = np.max(predict,axis=-1)

            #Prepare to get acc for answers in each class
            ind_cl = []
            y_cl = []
            for i in range(len(target)):
                ind_cl.append(ans==i)
                y_cl.append(y_k==i)
                
            fuzDom = {}
            threshold = -float('inf')
            mark = ans==y_k
            accs_cl = []
            for i in range(len(target)):
                chosen_in_cl = ind_cl[i]
                if np.sum(chosen_in_cl) > 0:
                    mark_cl = ans[chosen_in_cl]==y_k[chosen_in_cl]
                    accs_cl.append(np.sum(mark_cl)/len(mark_cl))    
                else:
                    accs_cl.append(float('nan'))
            # Below is a tuple of three entries.  
            # The first consists of overall acc and coverage rate.
            # The second consists of acc for each class.
            # The third consists of coverage rate for each class.
            fuzDom[threshold] = ((np.sum(mark)/len(mark),len(mark)/len(y_k)),
                                 np.array(accs_cl),
                                 np.full(len(target),1.))
            
            threshold = iniTh
            l = len(y_k)
            
            while l > minValNo:               
                req_certainty = sigmoid(threshold)
                certain = certainty>req_certainty
                l = np.sum(certain)
                if l>minValNo:                 
                    mark = ans[certain]==y_k[certain]
                    accs_cl = []
                    covrate_cl = []  #This remembers the covering rate for y==i in certain
                    for i in range(len(target)):
                        if np.sum(y_cl[i])>0:
                            covrate_cl.append(np.sum(certain * y_cl[i]) / np.sum(y_cl[i]))
                        else:
                            covrate_cl.append(0.)
                        chosen_in_cl = certain * ind_cl[i] == 1
                        if np.sum(chosen_in_cl) > 0:
                            mark_cl = ans[chosen_in_cl]==y_k[chosen_in_cl]
                            accs_cl.append(np.sum(mark_cl)/len(mark_cl))
                        else:
                            accs_cl.append(float('nan'))
                        
                    fuzDom[threshold] = ((np.sum(mark)/len(mark),len(mark)/len(y_k)),
                                         np.array(accs_cl),
                                         np.array(covrate_cl))
                threshold += step
            
            fuzDoms[k] = fuzDom
            if save:
                self.charts[k]['fuzDom'] = fuzDom
            if save and autosave:
                self.save()
            
        print("The fuzzy domains have been computed.") 
        return    

    def numActive(self):
        return len(self.activeKeys())

    def remove(self,keys,accBelow : float = None, write_story:bool=True):
        count = 0
        is_it_singleton = False
        removed_keys = []
        if not isinstance(keys,list):
            keys = [keys]
            is_it_singleton = True
        for k in keys:
            if accBelow is not None and self.charts[k]['fuzDom'][-float('inf')][0][0] > accBelow:
                continue
            filetype = self.charts[k]['filetype']
            path = self.modelPath(k,filetype)
            if os.path.isfile(path):
                os.remove(path)
            self.charts.pop(k)
            removed_keys.append(k)
            count += 1
        storyFile = self.storyFile
        if write_story and storyFile is not None:
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---Model Removed---\n")
                if is_it_singleton:
                    story.write(f"model {keys} has been removed from society ")
                    if accBelow is not None:
                        story.write(f"because its accuracy <= {accBelow}")
                    story.write(".\n")
                else:
                    story.write(f'{count} number of models have been removed from society.\n : {removed_keys}\n')
                    if accBelow is not None:
                        story.write(f"Their accuracy <= {accBelow}.\n")                    
        return removed_keys
    
    def removeInactive(self,storyFile : str = None):           
        removed_keys=self.remove([k for k in self.inactiveKeys()],write_story=False)
        count = len(removed_keys)
        if storyFile is not None:
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---Inactive Models Removed---\n") 
                story.write(f'{count} number of models have been removed from society.\n : {removed_keys}\n')
        return

    def groupByTarget(self, keys = None, active=True):
        groups = {}
        if keys is None:
            keys = self.charts.keys()
        for k in keys:
            if active and not self.charts[k]['active']:
                continue
            target = self.charts[k]['target']
            targetTuple = tuple([tuple(t) for t in target])
            if targetTuple not in groups:
                groups[targetTuple] = []                
            groups[targetTuple].append(k)
        return groups

    def keysInFlatTarget(self, flatTarget, keys=None, active=False, fine=False):
        if keys is None:
            keys = self.keys()
        chosen = []
        for k in keys:
            if active and not self.charts[k]['active']:
                continue
            if fine and not self.isFine(k):
                continue
            if flatTuple(self.charts[k]['target'])==flatTarget:
                chosen.append(k)
        return chosen    
    
    def keysInTarget(self, target, keys=None, active=False):
        if keys is None:
            keys = self.keys()
        chosen = []
        for k in keys:
            if active and not self.charts[k]['active']:
                continue
            if self.charts[k]['target']==target:
                chosen.append(k)
        return chosen
    
    #We include evaluate function into `predict'.
    #Evaluate if `y' is not None
    def predict(self, x, wantAcc=None, maskInWantAcc=None, keys=None, 
                batch_size : int =128,force_batch : bool =False, 
                active: bool =True, verbose : int =0,
                voteBy='weighAcc', onlyFineCanVote : bool =False,
                targetTree=None, keysTree=None, node=(),
                fullAns=None,certPart=None,originalAns=None,
                pred=None, pred_useHist=None, certs=None,
                out=None, modelAcc=None, # y : for evaluation only
                y=None, reportSeq=None, predOutputFile : str =None,
                evalOutputFile : str =None,
                show_av_acc : bool =False,show_simple_vote : bool =False, 
                count=0, useHistory=None, write_story : bool = True,
                display_ : bool = True):
        assert voteBy=='weighAcc' or voteBy=='order'
        if useHistory is not None and evalOutputFile is not None:
            assert useHistory != evalOutputFile
        # Initializing pred, certs, out, modelAcc, originalAns, wantAcc, and checking sanity of wantAcc
        if pred is None: pred = []
        if certs is None: certs = []
        if out is None: out = []
        if modelAcc is None: modelAcc = {}
        if originalAns is None:originalAns = {}
        if wantAcc is None:
            wantAcc = [0.5,0.7310585786300049,0.8807970779778823,0.9525741268224334,0.9820137900379085,0.9933071490757153,0.9975273768433653,0.9990889488055994,0.9996646498695336,0.9998766054240137]
            # The above list of floating numbers comes from list(sigmoid(np.arange(0, 10., 1.)))
        assert isinstance(wantAcc,(list,np.ndarray,float))
        if isinstance(wantAcc,float):
            wantAcc = [wantAcc]
        if node==():
            wantAcc = np.array(wantAcc)
            if wantAcc[0] != 0.:
                #We augment by wantAcc=0. in case some data are uncertain for all given wantAcc
                wantAcc = np.insert(wantAcc,0,0.)
        
        if targetTree is None or keysTree is None:
            if keys is None:
                keys=self.keys(active=active)               
            targetTree, keysTree = self.makeTargetTree(keys=keys,active=active)
        if maskInWantAcc is None:
            assert node==()
            maskInWantAcc = [np.full(len(x), True) for w in wantAcc]
        
        if y is not None:
            assert len(y) == len(x)
            if len(y.shape)==2:
                y = np.argmax(y,axis=-1) 
        
        if reportSeq is None:
            reportSeq = [50]
            while reportSeq[-1] < len(keys):
                reportSeq.append(reportSeq[-1] * 2)
        reportSeq = np.array(reportSeq)

        flatTarget = targetTree[node]
        cur_keys = keysTree[node]
        cur_keys = self.orderByAcc(cur_keys,flatTarget=flatTarget,fine=False)

        #Only pass to models when there is data needed to be processed
        needToProcess = [np.sum(mask) > 0 for mask in maskInWantAcc]
        if np.sum(needToProcess)==0:
            return fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist
        
        augReportSeq = np.insert(reportSeq,0,0)
        keys_gps = [cur_keys[augReportSeq[i]:augReportSeq[i+1]] for i in range(len(augReportSeq)-1) 
                    if augReportSeq[i] < len(cur_keys)]
        
        locCount=0 
        # locCount counts the number of models contained in a group while running the below for loop.

        for gp in keys_gps:
                   
            #Use models in keys_gps to predict. 
            thisAns,thisCertPart,thisOriginalAns = self.predictSameFlatTarget(x, wantAcc=wantAcc[needToProcess], 
                                                                   keys=gp, batch_size=batch_size,
                                                                   force_batch=force_batch, 
                                                                   onlyFineCanVote=onlyFineCanVote, 
                                                                   active=False, verbose=verbose,
                                                                   voteBy=voteBy)
            
            
            originalAns.update(thisOriginalAns)
            thisCertPart[0][:] = True
            # curAns, curCertPart <- Answer and certain part from current node.
            if node==():
                # Copy
                curAns = [a.copy() for a in thisAns]
                curCertPart = [c.copy() for c in thisCertPart]
            else:
                #Insert None padding to thisAns and thisCertPart
                curAns=[]
                curCertPart=[]
                i=0
                for need in needToProcess:
                    if need:
                        curAns.append(thisAns[i].copy())
                        curCertPart.append(thisCertPart[i].copy())
                        i+=1
                    else: # need == False, meaning that it has no masking.
                        curAns.append(np.zeros((len(x),len(flatTarget)), dtype = float))
                        # 0.0 instead of None
                        curCertPart.append(np.full(len(x),False,dtype=bool))
                        # False instead of None
            
            #Collect the answers at this node of targetTree
            #Modify by curAns and curCertPart
            if locCount==0: # local count counts how many loop in key gps we see, which is the case if reportSeq is less then the length of key group.
                ansAtNode = [a.copy() for a in curAns]
                certAtNode = [a.copy() for a in curCertPart]
            else:               
                for w in range(len(wantAcc)):
                    ansAtNode[w][curCertPart[w]*~certAtNode[w]==1] = curAns[w][curCertPart[w]*~certAtNode[w]==1]
                    
                    certAtNode[w] = certAtNode[w]+curCertPart[w]>0

            if node==():
                reportAns = []
                for a in ansAtNode:
                    if a is None:
                        reportAns.append(None)
                    else:
                        reportAns.append(a.copy())
                reportCert = []
                for a in certAtNode:
                    if a is None:
                        reportCert.append(None)
                    else:
                        reportCert.append(a.copy())
            else:
                reportAns = [a.copy() for a in fullAns]
                reportCert = [a.copy() for a in certPart]
                for w in range(len(wantAcc)):                    
                    #Modify the previous reportAns by the current node
                    #Since fancy indexing of numpy makes another copy and does not work for
                    #modifying the array, I make the following way using temp.                    
                    if voteBy=='weighAcc':       
                        #First, we modify by the certain part of the current node
                        care = maskInWantAcc[w] * certAtNode[w] > 0    
                        temp = reportAns[w][care]
                        temp[:,flatTarget] += ansAtNode[w][care]
                        flatTargetc = np.delete(np.array(range(self.target)),flatTarget) # the complement of flat target
                        temp[:,flatTargetc] = 0.
                        reportAns[w][care] = temp
                        
                        #Next, we modify the previous uncertain part by the uncertain part of the current node
                        care = maskInWantAcc[w] * ~certAtNode[w] * ~reportCert[w] > 0 
                        temp = reportAns[w][care]
                        temp[:,flatTarget] += ansAtNode[w][care]
                        flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                        temp[:,flatTargetc] = 0.
                        reportAns[w][care] = temp                    
                    else: #voteBy=='order'
                        #First, we modify by the certain part of the current node
                        care = maskInWantAcc[w] * certAtNode[w] > 0    
                        temp = reportAns[w][care]
                        temp[:,flatTarget] = ansAtNode[w][care]
                        flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                        temp[:,flatTargetc] = 0.
                        reportAns[w][care] = temp
                        
                        #Next, we modify the previous uncertain part by the uncertain part of the current node
                        care = maskInWantAcc[w] * ~certAtNode[w] * ~reportCert[w] > 0 
                        temp = reportAns[w][care]
                        temp[:,flatTarget] = ansAtNode[w][care]
                        flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                        temp[:,flatTargetc] = 0.
                        reportAns[w][care] = temp                        
                    #Finally, we expand the certain part by that of the current node
                    care = maskInWantAcc[w]   
                    reportCert[w][care] = reportCert[w][care] + certAtNode[w][care] > 0                    
     
            #We report the prediction for every node of targetTree and at each term of reportSeq                       
            pred.append([np.argmax(reportAns[w],axis=-1) for w in range(len(wantAcc))])
            certs.append(reportCert.copy())
            if predOutputFile is not None:
                predOut={}
                for w in range(len(wantAcc)):
                    predOut['Prediction at wantAcc %f'%wantAcc[w]] = pred[-1][w].copy()
                    predOut['Certain at wantAcc %f'%wantAcc[w]] = reportCert[w].copy()
                predOut = pd.DataFrame(predOut)
                predOut.to_csv(self.path+predOutputFile+'_%02d.csv'%(len(pred)-1),index=False)
            #If useHistory, then produce pred_useHist in csv file by
            #choosing the prediction for each x with the best expected acc 
            #(according to its certainty and the history for validation data).
            #useHistory stores the name of history (for instance the name of committee)
            #The history we use is the acc in certainty domain according to 
            #the certainty levels asserted by wantAcc array.
            #pred_useHist is a list with three entries:
            #pred_useHist[0] is storing predictions in all stages,
            #pred_useHist[1] is storing expected accuracies in all stages.
            #pred_useHist[2] is a list of dataframes that analyze the changes in expected acc.
            #Both entries are numpy array with shape (len(stages),len(y))
            if useHistory is not None:
                if len(pred)==1:
                    pred_useHist = [np.empty((0,len(x))),np.empty((0,len(x))),[]]
                    
                accs_record = []
                bestwantAcc_record = []
                for i in range(len(pred)): #run over each page
                    hist = pd.read_csv(self.path + useHistory + '_%02d.csv'%i)                    
                    accs = np.array(hist['acc by refined vote restricted on confident domain'])
                    #cert_transpose whether the sample point is classified in certain or not in wanted accs.
                    #It has shape (len(x),len(wantAcc))                        
                    cert_transpose = np.transpose(np.array(certs[i]))
                    masked = np.ma.masked_array(np.repeat(np.expand_dims(accs,axis=0),
                                                              len(x),axis=0),
                                                mask=~cert_transpose,fill_value=0.)
                    #accs_x records the largest expected acc with certainty that each sample point
                    #can achieve.  It has shape (len(x),)
                    #bestwantAcc_x records the best wantAcc to maximize the expected acc
                    #for each x.                      
                    accs_x = masked.max(axis=-1)
                    bestwantAcc_x = masked.argmax(axis=-1)
                    accs_record.append(accs_x)
                    bestwantAcc_record.append(bestwantAcc_x)
                #Make both accs_record and bestwantAcc_record to np array (len(x),len(pred))
                accs_record = np.transpose(np.array(accs_record)) 
                bestwantAcc_record = np.transpose(np.array(bestwantAcc_record))
                #find in which page pred has the best acc
                #best_page has shape (len(x),1)
                best_page = accs_record.argmax(axis=-1,keepdims=True)
                bestwantAccinbest_page = np.take_along_axis(bestwantAcc_record, 
                                                            best_page, axis=1).astype(int)
                pred_np = np.transpose(np.array(pred))
                #pred_np has shape (len(x),len(wantAcc),len(pred))
            
                #First match the shape by repeating best_page to (len(x),len(wantAcc),len(pred))
                best_page = np.repeat(np.expand_dims(best_page,axis=1),len(wantAcc),axis=1)

                
                best_pred = np.take_along_axis(pred_np, best_page, axis=2)
                best_pred = np.squeeze(best_pred, axis=2)
                best_pred = np.take_along_axis(best_pred, bestwantAccinbest_page, 
                                               axis=1).astype(int)
                
                #best_pred has shape (len(x),1)
                #pred_useHist[i] has shape (?,len(x))
                #Need to match the shapes
                pred_useHist[0] = np.append(pred_useHist[0],
                                            np.transpose(best_pred),
                                            axis=0)   
                pred_useHist[1] = np.append(pred_useHist[1],
                                            np.expand_dims(accs_record.max(axis=-1),axis=0),
                                            axis=0)
                
                if len(pred_useHist[1])>1:
                    #We make a dataframe to analyze the expected acc changes
                    #of pred_useHist from last batch of models.   
                              
                    classes_old, cl_assign_old = np.unique(pred_useHist[1][-2], return_inverse=True)
                    classes_new, cl_assign_new = np.unique(pred_useHist[1][-1], return_inverse=True)
                    #Then we compute the number of changes 
                    change_df = pd.DataFrame(columns=['Old acc', 'New acc', 'Number of data'])
                    df_type = {'Old acc': float, 'New acc': float, 'Number of data':int}                    
                    change_df = change_df.astype(df_type)
                    row=0
                    for c_old in classes_old:
                        for c_new in classes_new:
                            num_changes = np.sum(classes_new[cl_assign_new[classes_old[cl_assign_old] 
                                                                           == c_old]] 
                                                 == c_new).astype(int)
                            if num_changes>0:
                                change_df.loc[row] = [c_old,c_new,num_changes]                        
                                row+=1
                    pred_useHist[2].append(change_df)
                    if verbose>0:
                        print("Change in expected acc of prediction using history:")
                        display(change_df)                    
                
                if predOutputFile is not None:
                    pred_useHistOut={}
                    pred_useHistOut['Predictions'] = pred_useHist[0][-1]
                    pred_useHistOut['Expected accuracy'] = pred_useHist[1][-1]
                    pred_useHistOut = pd.DataFrame(pred_useHistOut)
                    pred_useHistOut.to_csv(self.path+predOutputFile+'_useHist_%02d.csv'%(len(pred_useHist[0])-1),index=False)
                    change_df.to_csv(self.path+predOutputFile+'_accChangesUseHist_%02d.csv'%(len(pred_useHist[2])),index=False)
            
            #Then we shorten reportSeq accordingly                
            if len(gp) < reportSeq[0]:
                reportSeq[0] -= len(gp)    
            else:
                reportSeq = np.delete(reportSeq,0)   
            count += len(gp)
            locCount += len(gp)
            
            #For evaluation only
            if y is not None:
                #Refined vote using certainty               
                acc_full = []
                acc_in_certain = []
                size_in_certain = []
                for w in range(len(wantAcc)):
                    acc_full.append(np.sum(pred[-1][w] == y) / len(y))
                    size = np.sum(reportCert[w])
                    size_in_certain.append(size)
                    if size>0:
                        cer_ans = pred[-1][w][reportCert[w]]
                        acc_in_certain.append(np.sum(cer_ans == y[reportCert[w]]) / len(cer_ans))
                    else:
                        acc_in_certain.append(0.)
                    #relaxed_check = np.full(len(y),False)
                    #for i in range(len(ans)):
                    #    pos_ans = (np.argwhere(ans[i] == np.max(ans[i]))).flatten()
                    #    relaxed_check[i] = y[i] in pos_ans 
                    #    
                    #cer_relaxed_check = np.full(len(cer_ans),False)
                    #for i in range(len(cer_ans)):
                    #    pos_ans = (np.argwhere(cer_ans[i] == np.max(cer_ans[i]))).flatten()
                    #    cer_relaxed_check[i] = y[cerPart][i] in pos_ans                
                show = {}
                if node == () and (show_av_acc or show_simple_vote):
                    if len(originalAns)>0:
                        for k in cur_keys:
                            if k in originalAns:
                                modelAcc[k] = np.sum(np.argmax(originalAns[k],axis=-1) == y)/len(y)  
                        originalAns_arr = np.array(list(originalAns.values()))      
                        #Just average originalAns
                        averageAns = np.argmax(np.mean(originalAns_arr,axis=0),axis=-1)            
                        #Simple vote
                        vote = np.argmax(np.sum(tf.one_hot(np.argmax(originalAns_arr,axis=-1),
                                                           self.target).numpy(),axis=0),axis=-1)
                        av_acc = np.sum(averageAns==y)/len(y)
                        vote_acc = np.sum(vote==y)/len(y)                                                
                        if show_av_acc:
                            show['acc by taking average'] = av_acc
                        if show_simple_vote:
                            show['acc by simple vote'] = vote_acc
                show['acc by refined vote'] = acc_full
                show['acc by refined vote restricted on confident domain'] = acc_in_certain
                show['size of certain part'] = size_in_certain
                # index = -np.log(1/wantAcc - 1)
                index = wantAcc

                if len(out)>0:
                    #We make a dataframe to analyze the acc changes
                    #from last batch of models.   
                    
                    #accs record the actual acc with certainty in wanted accs.
                    #It has shape (len(wantAcc),)
                    accs_old = np.array(out[-1][0]['acc by refined vote restricted on confident domain'])
                    accs_new = np.array(show['acc by refined vote restricted on confident domain']) 
                    
                    #cert record whether the sample point is classified in certain or not in wanted accs.
                    #It has shape (len(y),len(wantAcc))
                    cert_old = np.transpose(np.array(certs[-2]))                                      
                    cert_new = np.transpose(np.array(certs[-1])) 
                    #acc_cl records the largest acc with certainty that each sample point
                    #can achieve.  It has shape (len(y),)
                    #acc_cl indeed records the class indices (coming from wantAcc)
                    #rather than the actual acc.
                    masked_old = np.ma.masked_array(np.repeat(np.expand_dims(accs_old,axis=0),
                                                 len(y),axis=0),
                                       mask=~cert_old,fill_value=-float('inf'))
                    acc_cl_old = masked_old.argmax(axis=-1)
                    masked_new = np.ma.masked_array(np.repeat(np.expand_dims(accs_new,axis=0),
                                                 len(y),axis=0),
                                       mask=~cert_new,fill_value=-float('inf'))
                    acc_cl_new = masked_new.argmax(axis=-1) 
                    #cl_assign has shape (len(y),)
                    #It records the classes (coming from wantAcc) that each sample point belongs to.
                    classes_old, cl_assign_old = np.unique(acc_cl_old, return_inverse=True)
                    classes_new, cl_assign_new = np.unique(acc_cl_new, return_inverse=True)
                    #Then we compute the number of changes 
                    change_df = pd.DataFrame(columns=['Old acc', 'New acc', 'Number of data'])
                    df_type = {'Old acc': float, 'New acc': float, 'Number of data':int}                    
                    change_df = change_df.astype(df_type)
                    row=0
                    for c_old in classes_old:
                        for c_new in classes_new:
                            num_changes = np.sum(classes_new[cl_assign_new[classes_old[cl_assign_old] 
                                                                           == c_old]] 
                                                 == c_new).astype(int)
                            if num_changes>0:
                                change_df.loc[row] = [accs_old[c_old],accs_new[c_new],num_changes]                        
                                row+=1                    
                    if useHistory is None:
                        out.append([pd.DataFrame(show,index=index),change_df,None])
                    else:
                        #pred_useHist[i] has shape (number of pages,len(x))
                        result = (pred_useHist[0]==y)*1.
                        finalAcc = np.sum(result,axis=-1) / len(y)
                        out_useHist = {'Accuracy':finalAcc}
                        out.append([pd.DataFrame(show,index=index),change_df,out_useHist])
                else:
                    if useHistory is None:
                        out.append([pd.DataFrame(show,index=index),None,None])  
                    else:
                        result = (pred_useHist[0]==y)*1.
                        finalAcc = np.sum(result,axis=-1) / len(y)
                        out_useHist = {'Accuracy':finalAcc}
                        out.append([pd.DataFrame(show,index=index),None,out_useHist])
                if verbose>0:
                    print("Current target of models is:", flatTarget)
                    print("Result of the best %d models in the prescribed list of certainty levels:"%count)
                    if display_:
                        display(out[-1][0])
                    if out[-1][1] is not None:
                        print("Changes compared to the last page (for predictions without using history):")
                        display(out[-1][1])
                    if out[-1][2] is not None:
                        print("Results using expected accuracy from past history:")
                        display(out[-1][2])
                if evalOutputFile is not None:
                    out[-1][0].to_csv(self.path+evalOutputFile+'_%02d.csv'%(len(out)-1))
                    if out[-1][1] is not None:
                        out[-1][1].to_csv(self.path+evalOutputFile+'_change_%02d.csv'%(len(out)-1))
                    if out[-1][2] is not None:
                        out[-1][2].to_csv(self.path+evalOutputFile+'_useHist_%02d.csv'%(len(out)-1))
                                       
        if node==():
            fullAns = ansAtNode.copy()
            certPart = certAtNode.copy()
        else:
            for w in range(len(wantAcc)):            
                #Modify the previous fullAns by the current node
                #Since fancy indexing of numpy makes another copy and does not work for
                #modifying the array, I make the following way using temp.                    
                if voteBy=='weighAcc':       
                    #First, we modify by the certain part of the current node
                    care = maskInWantAcc[w] * certAtNode[w] > 0    
                    temp = fullAns[w][care]
                    temp[:,flatTarget] += ansAtNode[w][care]
                    flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                    temp[:,flatTargetc] = 0.
                    fullAns[w][care] = temp
                    
                    #Next, we modify the previous uncertain part by the uncertain part of the current node
                    care = maskInWantAcc[w] * ~certAtNode[w] * ~certPart[w] > 0 
                    temp = fullAns[w][care]
                    temp[:,flatTarget] += ansAtNode[w][care]
                    flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                    temp[:,flatTargetc] = 0.
                    fullAns[w][care] = temp                    
                else: #voteBy=='order'
                    #First, we modify by the certain part of the current node
                    care = maskInWantAcc[w] * certAtNode[w] > 0    
                    temp = fullAns[w][care]
                    temp[:,flatTarget] = ansAtNode[w][care]
                    flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                    temp[:,flatTargetc] = 0.
                    fullAns[w][care] = temp
                    
                    #Next, we modify the previous uncertain part by the uncertain part of the current node
                    care = maskInWantAcc[w] * ~certAtNode[w] * ~certPart[w] > 0 
                    temp = fullAns[w][care]
                    temp[:,flatTarget] = ansAtNode[w][care]
                    flatTargetc = np.delete(np.array(range(self.target)),flatTarget)
                    temp[:,flatTargetc] = 0.
                    fullAns[w][care] = temp                        
                #Finally, we expand the certain part by that of the current node
                care = maskInWantAcc[w]   
                certPart[w][care] = certPart[w][care] + certAtNode[w][care] > 0                
        
        nextNodes = [n for n in keysTree.keys() if n[:-1]==node and len(n)!=0]
        def maxacc(n):
            return np.max([self.acc(k) for k in keysTree[n]])
        nextNodes.sort(key=maxacc,reverse=True)
        for n in nextNodes:
            #pass those x predicted to be inside the target of n to the models at n
            #Since we do not call model.predict repeatedly, we pass the whole x to the next models,
            #And pick those answers that are in target of n later. 
            #maskInWantAcc records the data needed to be processed at current node.
            #Also, n in nextNodes are ordered by acc.
            #If samples already taken by previous n in nextNodes, then don't take it for later n in nextNodes.
            #This is also recorded in maskInWantAcc.
            newTarget = targetTree[n]
            matchTarget = []
            for w in range(len(wantAcc)):
                
                matchTarget.append(2* np.sum(fullAns[w][:,newTarget],axis=-1) 
                                   >= np.sum(fullAns[w],axis=-1))          
            thisMaskInWantAcc = [maskInWantAcc[w] * matchTarget[w] > 0
                                for w in range(len(wantAcc))]
            # Searching for prediction in target tree.
            # Here is an example.
            '''
            Target Tree
            T_0 = full and fine target.
            tree : 
            T_0 -> T_1 -> T_4
                -> T_2
                -> T_3
            where arrows mean "contatining"
            
            "Mask" masks input as 'it may require more votes from model with certainty'. Initially it is True at all input position.
            Let a position be fixed.
            MatchTarget is False if, given that length of new targets is not less than 2,
            (sum of accuracies to new targets/length of new targets) < (sum of accuracies from parent node /length of parent targets)
            (note that the shorter length of new target is the more specialized targets)
            , which means even though we could have more specialist for new targets parent answer acertain their prediction.
            Shortly, we don't pass input to sub-prediction when we made sure of that prediction already.
            
            T_1, T_2, T_3 are ordered by accuracies.
            
            Recursively we mask off initial mask by searching our tree in depth.
            
            Shortly, full answer is modified by new answer unless pre-certain and cur-uncertain.
            cert part will be updated as recursion goes.
            
            As  T_1->T_4, we have updated fullAns, certpart, pred and certs(they are just full Ans, certPart if reportSeq is sufficiently big)
            out : table for result, 
            modelAcc : recording usual accuracy while collecting votes from models we have met
            
            And pass the results to the next node, T_2.
            Basically it is depth-first search.
            '''
            fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist = self.predict(x,
                wantAcc=wantAcc, 
                batch_size=batch_size, force_batch=force_batch,
                maskInWantAcc=thisMaskInWantAcc, 
                node=n,
                targetTree=targetTree, 
                keysTree=keysTree,
                fullAns=fullAns,
                certPart=certPart,
                originalAns=originalAns,
                pred = pred, 
                pred_useHist=pred_useHist,
                certs = certs, 
                out=out, 
                modelAcc = modelAcc,
                verbose=verbose,
                onlyFineCanVote=onlyFineCanVote,
                y=y,
                reportSeq=reportSeq, 
                predOutputFile=predOutputFile, 
                evalOutputFile=evalOutputFile,
                show_av_acc=False,show_simple_vote=False, #taking average or counting make sense with full & fine targets 
                count=count, # "count" counts the number of models involved in one prediction table.
                voteBy=voteBy,useHistory=useHistory,
                write_story=False)

            
            for w in range(len(wantAcc)):
                maskInWantAcc[w] = maskInWantAcc[w] * (~matchTarget[w]) > 0
        
        if node==():
            storyFile = self.storyFile
            #Only write into story for evaluation
            if storyFile is not None and write_story and y is not None:
                with open(self.path+storyFile+".txt", "a") as story:
                    story.write('---Society Evaluated---\n')
                    story.write(f'{len(keys)} number of models are involved.  Data size is {len(y)}.\n')
                    story.write(f'Models = {keys}')
                    story.write('Average acc by refined vote in pages without using history:\n')
                    for i in range(len(out)):
                        story.write('%.4f  '%(np.mean(out[i][0]['acc by refined vote'])))
                    story.write('\n')
                    if useHistory is not None:
                        story.write(f'History in filename {useHistory} is now used in prediction.\n')
                        story.write('Accuracy of predictions using history in pages:\n')
                        for a in out[-1][2]['Accuracy']:
                            story.write('%.4f  '%a)
                        story.write('\n')
                    if evalOutputFile is not None:                    
                        story.write(f'Evaluation result saved as {evalOutputFile}.\n')
                    if predOutputFile is not None:
                        story.write(f'Prediction result saved as {predOutputFile}.\n')
        return fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist

    def predictSameFlatTarget(self, x, wantAcc, keys=None, batch_size=128, force_batch=False, 
                onlyFineCanVote=False, active=True, verbose=1,
                voteBy='weighAcc'):
        # Predict by counting votes.
        # return fullAns as a matrix (sample,vote count vector) 
        # and certPart (a mask for samples that there exists a model being certain about it)
        # and original answer
        # Assume all models here have the same flat target (ex. (2,3,4))
        # Can vote by 'weighAcc' or 'order'
        # Currently, only consider fine models if vote by 'order'
        
        if keys is None:
            keys=self.keys()
        groups = self.groupByFlatTarget(keys=keys,active=active)
        assert len(list(groups.keys()))==1
        flatTarget = list(groups.keys())[0]
        lenTarget = len(flatTarget)
        keys = groups[flatTarget]

        if not isinstance(wantAcc,(list,np.ndarray)):
            wantAcc = [wantAcc]
        if len(wantAcc)==0:
            return None,None,{}
            
        originalAns = {}     
        assert voteBy=='weighAcc' or voteBy=='order'
        if voteBy=='order':
            #Only accept vote from fine model for the moment
            groups = self.groupByTarget(keys=keys,active=active)
            targetTuple = tuple([(i,) for i in flatTarget])
            keys = self.orderByAcc(groups[targetTuple],flatTarget=flatTarget)
            
            model = self.getModel(keys[0])
            cur_batch_size = batch_size
            if model.count_params() > 1e7: 
                cur_batch_size = 32
            else:
                cur_batch_size = batch_size                
            if force_batch == True:
                if verbose==1:
                    print("%s is thinking..."%self.modelName(keys[0]))
                pred = self.force_batch_model_predict(model, x, cur_batch_size)      
            else:
                pred = model.predict(x,batch_size=cur_batch_size,verbose=0)
            assert len(pred.shape)==2                        
            originalAns[keys[0]] = pred                
            fullAns = [pred for w in wantAcc]
            
            certPart = []
            fuzDom = self.charts[keys[0]]['fuzDom']
            certainty = np.max(pred,axis=-1)                                       
            for w in range(len(wantAcc)):
                pos = np.array([th for th in fuzDom.keys() 
                            if fuzDom[th][0][0] >= wantAcc[w]])
                # Get thresholds greater than (possibly augmented by 0 at the first index) wanted accuracy rates
                if len(pos)==0:
                    certPart.append(np.full(len(x),False))                      
                else:
                    threshold = np.min(pos)
                    certain = certainty>sigmoid(threshold)
                    #Only those answered in thin classes can be regarded as certain
                    certPart.append(certain)                                             
            
            # certPart is an array of length |wantAcc| possibly augmented 
            # consisting of boolean values of size x(input data) 
            # from model with highest accuracy rate.
            # np.sum(certPart[w]) is decreasing seq. in w.
            # 'True' means the input data is certain by the highest accuracy model.
            for key in keys[1:]:
                model = self.getModel(key)
                fuzDom = self.charts[key]['fuzDom']
                if verbose==1:
                    print("%s is thinking..."%self.modelName(key))
                if model.count_params() > 1e7:
                    cur_batch_size = 32
                else:
                    cur_batch_size = batch_size
                if force_batch == True:
                    pred = self.force_batch_model_predict(model, x, cur_batch_size)      
                else:
                    pred = model.predict(x,batch_size=cur_batch_size,verbose=0)
                originalAns[key] = pred       
                
                #Get certain part       
                assert len(pred.shape)==2                                      
                certainty = np.max(pred,axis=-1) 

                for w in range(len(wantAcc)):
                    pos = np.array([th for th in fuzDom.keys() 
                                if fuzDom[th][0][0] >= wantAcc[w]])
                    
                    if len(pos)>0:
                        threshold = np.min(pos)
                        certain = certainty>sigmoid(threshold)
                        fullAns[w][certain*~certPart[w]==1] = pred[certain*~certPart[w]==1]
                        # fullAns is consisting of predictions by highest accuracy model.
                        # certain = certain data of model[key]
                        # certPart[w] = certain data with w-wantAcc from highest accuracy model.   
                        # fullAns[w] record certain data which is not certain in highest accuracy model.         
                        certPart[w] = certain+certPart[w]>0
                        # And then update certPart at w.
        elif voteBy=='weighAcc':
            keys = self.orderByAcc(keys,flatTarget=flatTarget,fine=onlyFineCanVote)
            #Make a list of eachCertPart
            eachCertPart = [np.full((len(x),len(keys)),False) for want in wantAcc]
            eachAns = [np.full((len(x),len(keys),lenTarget),0.) for want in wantAcc]     
                    
            for i in range(len(keys)):
                target = self.charts[keys[i]]['target']
                fuzDom = self.charts[keys[i]]['fuzDom']
                assert fuzDom != None

                model = self.getModel(keys[i])
                if model.count_params() > 1e7:
                    cur_batch_size = 32        
                else:
                    cur_batch_size = batch_size
                if verbose==1:
                    print("%s is thinking..."%self.modelName(keys[i]))
                if force_batch == True:
                    pred = self.force_batch_model_predict(model, x, cur_batch_size)      
                else:
                    pred = model.predict(x,batch_size=cur_batch_size,verbose=0)
                
                #Get certainty and model answers           
                assert len(pred.shape)<=2                        
                if len(pred.shape) == 2:
                    #only record original answers for complete models
                    if len(target) == self.target:
                        originalAns[keys[i]] = pred                
                    certainty = np.max(pred,axis=-1)
                    modelAns = np.argmax(pred,axis=-1)
                else: #len(pred.shape) == 1
                    assert len(target)==2 
                    if len(target) == self.target:
                        originalAns.append(tf.one_hot(pred,self.target).numpy())                 
                    certainty = pred
                    modelAns = (pred > 0.5) * 1                            
                    
                #Find the part in certain.  Recorded in eachCertPart.
                #Also record the acc per class.
                #If target of model has `thick class', ex. target=[[0],[1,2]], then
                #only those answered in thin classes can be regarded as certain.
                #Also, in thick case, we require acc in each class reaches required level.
                
                width = np.array([len(t) for t in target])
                #assert np.min(width)==1
                thin_cl = np.where(width==1)[0]            
                #inThin valued in T/F records those samples whose modelAns are in thin classes
                if len(thin_cl)>0:
                    inThin = np.sum(np.array([modelAns == c for c in thin_cl]),axis=0)>0
                else:
                    inThin = np.full(len(modelAns),False)
    
                acc_cl = []
                acc_cl_in_certain = []
                for w in range(len(wantAcc)):
                    if len(target) < lenTarget:
                        pos = np.array([th for th in fuzDom.keys() 
                                    if np.nanmin(fuzDom[th][1]) >= wantAcc[w]])
                    else:
                        pos = np.array([th for th in fuzDom.keys() 
                                        if fuzDom[th][0][0] >= wantAcc[w]])
                    if len(pos)==0:
                        #print('No certainty level found to reach the required acc %.2f'%wantAcc[w])
                        acc_cl.append(fuzDom[-float('inf')][1])
                        acc_cl_in_certain.append(np.zeros(len(target)))                        
                    else:
                        threshold = np.min(pos)
                        certain = certainty>sigmoid(threshold)
                        #Only those answered in thin classes can be regarded as certain
                        eachCertPart[w][:,i] = certain*inThin==1
                        acc_cl.append(fuzDom[-float('inf')][1])
                        acc_cl_in_certain.append(fuzDom[threshold][1])
    
                # Now collect answer from each model.
                # Assume target is a list of lists of integers>=0
                # ex. target=[[2],[1,0]] or [[0],[1],[2]].
                # The first means label to each sample is determined as 2,
                # or determined as belonging to {1,0}.
    
                #ex. target=[[2],[1,0]]
                #target_vec = [[0,0,1],[0.5,0.5,0]]
                target_vec = np.array([tf.reduce_mean(tf.one_hot(t,lenTarget),axis=0).numpy() 
                                           for t in reIndex(target)])
                #ex. modelAns=[1,1,0]
                #Then model_Ans_oneHot = [[0,1],[0,1],[1,0]]
                modelAns_oneHot = tf.one_hot(modelAns,
                                              len(target),off_value=0).numpy() * 1.
                #Alternatively, we can take
                #model_Ans_oneHot = [[-1,1],[-1,1],[1,-1]].
                #This distinguish the answer from not voting (zero).
                #On the other hand, all models summed over here have the same flat target.
                #Thus don't need to worry about distinguishing from zero.
                #Being in [0,1] has better probabilistic interpretation.
                #modelAns_oneHot = tf.one_hot(modelAns,
                                              #len(target),off_value=-1).numpy() * 1.                
                
                #Next, we put weight on each answer according to validation accuracy of each model.
                for w in range(len(wantAcc)):
                    certain = eachCertPart[w][:,i]
                    modelAns_oneHot_w = modelAns_oneHot.copy()
                    if np.sum(~certain)>0:
                        modelAns_oneHot_w[~certain] *= acc_cl[w][modelAns][~certain][:,np.newaxis]
                    if np.sum(certain)>0:
                        modelAns_oneHot_w[certain] *= acc_cl_in_certain[w][modelAns][certain][:,np.newaxis]
                    eachAns[w][:,i,:] = np.dot(modelAns_oneHot_w,target_vec)     
            
            print("Vote counting...") 
            certPart = []
            for w in range(len(wantAcc)):
                certPart.append(np.sum(eachCertPart[w],axis=-1)>0)
    
            #Determine the final answer by vote.
            #Each vote is valued in the inteval [0,1].
            #For certPart in sample, only take votes from answers with confidence.
            #For ~certPart in sample, take votes from all answers.
    
            fullAns = [np.zeros((len(x),lenTarget)) for want in wantAcc]
            for w in range(len(wantAcc)):
                if np.sum(~certPart[w])>0:
                    fullAns[w][~certPart[w]] = np.sum(eachAns[w][~certPart[w]], axis=1)                
                if np.sum(certPart[w])>0:
                    fullAns[w][certPart[w]] = np.sum(
                        eachCertPart[w][certPart[w]][...,np.newaxis] 
                        * eachAns[w][certPart[w]], axis=1)
                    #    fullAns[w][certPart[w]] = np.array([np.sum(eachAns[w][certPart[w]][j,eachCertPart[w][certPart[w]][j],:], axis=0) 
                    #                              for j in range(len(eachAns[w][certPart[w]]))])          
        
        return fullAns, certPart, originalAns
    
    def cover(self, k, wantAcc : float, batch_size : int =32):
        model = self.getModel(k)
        fuzDom = self.charts[k]['fuzDom']           
        pred = model.predict(self.x_v,batch_size=batch_size,verbose=0)
        
        #Get certainty and model answers           
        assert len(pred.shape)==2                                        
        certainty = np.max(pred,axis=-1)
        modelAns = np.argmax(pred,axis=-1) 
        correct = np.argmax(self.y_v,axis=-1)==modelAns
        
        pos = np.array([th for th in fuzDom.keys() 
                    if np.nanmin(fuzDom[th][0][0]) >= wantAcc])
        
        if len(pos)==0:
            return np.full(len(modelAns),False)
        
        threshold = np.min(pos)
        certain = certainty>sigmoid(threshold)
        # Return "certainly correct" in boolean valued array.        
        return correct * certain == 1        
    
    #Only do battle for fine models
    def battle(self, wantAcc : float, flatTarget : list =None, verbose : int =1):
        if flatTarget is None:
            target = [[i] for i in range(self.target)]
        else:
            target = [[i] for i in flatTarget]
        keys = [k for k in self.charts.keys() 
                if self.charts[k]['active'] and self.charts[k]['target']==target]
        keys = self.orderByAcc(keys)
        covered = np.full(len(self.y_v),False)
        count_deact = 0
        count_act = 0
        for k in keys:
            cover = self.cover(k,wantAcc)
            if np.sum(cover)==0: # if there is no certainly correct in wantAcc threshold
                self.deactivate(k)
                if verbose==1:
                    print(self.modelName(k) + " deactivated.")
                    count_deact +=1
                continue
            if np.prod(covered[cover])==1: 
                # without this model, its certainly correct part can be covered by other models having higher accuracy
                self.deactivate(k)
                if verbose==1:
                    print(self.modelName(k) + " deactivated.")     
                    count_deact +=1           
            else:
                covered = (covered + cover > 0)
                if verbose==1:
                    print(self.modelName(k) + " survived.")
                    count_act +=1
        storyFile = self.storyFile
        if storyFile is not None:
            with open(self.path+storyFile+".txt", "a") as story:
                story.write('---Battle Started---\n')
                story.write(f'{count_deact} number of models deactivated and {count_act} has survived. Wanted accuracy rate was set at {wantAcc})\n')                 
        return
    
    def findValDom(self, wantAcc : float , keys=None, active : bool =True,verbose : int =0,batch_size : int =32,
                   autosave=True):
        print("Finding domain of the logifold...")
        assert isinstance(wantAcc,float)
        if keys is None:
            keys = self.keys()
        if active:
            keys = [k for k in keys if self.charts[k]['active']]
        # fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist : output of predict.
        predictions_result = self.predict(self.x_v, wantAcc, keys=keys,
                                          active=active, verbose=verbose,batch_size=batch_size)
        fullAns, certPart = predictions_result[0:2]
        #Note self.predict always pretend wantAcc is a list and returns a list, we need to take [0] here
        modelAns = np.argmax(fullAns[0],axis=-1)  
        correct = np.argmax(self.y_v,axis=-1)==modelAns
        if wantAcc != 0.:
            index = 1
        else:
            index = 0
        certain = certPart[index] #recall that certPart is a mask consisting of True/False
        self.valDom = (correct)*(certain)==1
        print("Size of validation domain:%d"%(np.sum(self.valDom)))
        print("%d were not predicted correct and %d were uncertain."
              %(np.sum(~correct),np.sum(~certain)))
        if autosave:
            self.save()
        return self.valDom

    def keys(self, active=False, inactive=False):
        assert not (active and inactive)
        if active:
            return self.activeKeys()
        elif inactive:
            return self.inactiveKeys()
        else:
            return list(self.charts.keys())

    def activeKeys(self, keys=None):
        if keys is None:
            keys = self.charts.keys()
        return [k for k in keys if self.charts[k]['active']]
    
    def inactiveKeys(self, keys=None):
        if keys is None:
            keys = self.keys()        
        return [k for k in keys if not self.charts[k]['active']]
    
    def newLabelForTarget(self, x, y, target, digit:bool =False, dataType:str =None):
        if dataType is None:
            dataType = self.dataType
        newout = len(target)
        if dataType == 'numpy':            
            y_t = np.full(len(y),-1)
            assert len(y.shape)<=2
            if len(y.shape)==2:
                classOfIndices = [np.where(np.argmax(y,axis=-1)==t)[0] 
                             for t in range(self.target)]
            else:
                classOfIndices = [np.where(y==t)[0] for t in range(self.target)]          
          
            newClassOfIndices = [np.concatenate([classOfIndices[t] for t in c],axis=0) 
                             for c in target]
            for i in range(newout):
                y_t[newClassOfIndices[i]] = i         
            x = x[y_t != -1]
            y_t = y_t[y_t != -1]     
            
            if digit or len(y.shape)==1:
                return x, y_t
            y_t = keras.utils.to_categorical(y_t, newout)  
            return x, y_t
        elif dataType == 'datagen':
            def newgen(datagen):
                for batch in datagen:
                    y_t = np.full(len(batch[1]),-1)
                    classOfIndices = [np.where(np.argmax(batch[1],axis=-1)==t)[0] 
                                 for t in range(self.target)]
                    newClassOfIndices = [np.concatenate([classOfIndices[t] for t in c],axis=0) 
                             for c in target]
                    for i in range(newout):
                        y_t[newClassOfIndices[i]] = i
                    x_t = batch[0][y_t != -1]
                    y_t = y_t[y_t != -1]
                    y_t = keras.utils.to_categorical(y_t, newout)
                    yield x_t,y_t
            return newgen(x)
        elif dataType == 'ds':
            def filter_func(x,y):
                #assume y is a 1d onehot array
                flatTarget = flatTuple(target)
                return tf.math.reduce_any([tf.math.equal(tf.argmax(y,axis=-1),t) 
                                              for t in flatTarget],axis=0)
            x = x.filter(filter_func)    
                
            def map_func(x,y):
                y_t = tf.argmax([tf.math.reduce_any([tf.math.equal(tf.argmax(y,axis=-1),t) 
                                              for t in c],axis=0) 
                           for c in target],axis=-1)
                y_t = tf.one_hot(y_t, newout)                     
                return x, y_t
            return x.map(map_func)
        else:
            print('Data type not supported.')
            return
    
    def incWtsForTarget(self, x, y, classes, wt=1.5):
        y_width=len(y.shape)
        if y_width==1:
            num_cl = np.max(y)+1
        if y_width==2:
            num_cl = y.shape[1]
            y = np.argmax(y,axis=-1)
            
        ClassOfIndices = [np.where(y==i)[0] for i in range(num_cl)]
        size = np.max([len(c) for c in ClassOfIndices])
        wts = np.full(num_cl,1.)
        classes = np.array(classes)
        wts[classes] = wt
        sample_ind = [np.random.choice(ClassOfIndices[i], size=int(size*wts[i])) 
                       for i in range(num_cl)]
        x_new = np.concatenate([x[ind] for ind in sample_ind],axis=0)
        y_new = np.concatenate([y[ind] for ind in sample_ind],axis=0)  
        
        if y_width==2:
            y_new = keras.utils.to_categorical(y_new, num_cl)  
        return x_new, y_new
    
    def samplesForTarget(self, newTarget, data_x, data_y, newLabel=True):
        if newLabel:
            data_x, y_t = self.newLabelForTarget(data_x, data_y, newTarget)
        newClassOfIndices = [np.where(np.argmax(y_t,axis=-1)==i)[0] for i in range(len(newTarget))]
        size = np.max([len(c) for c in newClassOfIndices])
        
        #Then randomly select samples from each group to form a new training sample.
        #Old version: If there is image generator, then use it to enlarge smaller samples
        #if datagen is None:
        sample_ind = [np.random.choice(ind, size=size) 
                       for ind in newClassOfIndices]
        x_new = np.concatenate([data_x[ind] for ind in sample_ind],axis=0)
        y_new = np.concatenate([y_t[ind] for ind in sample_ind],axis=0)
        #else:
        #    x_new = []
        #    y_new = []
        #    for ind in newClassOfIndices:
        #        x_new.append(x[ind])
        #        y_new.append(y_t[ind])                
        #
        #        turn=0
        #        for batch in datagen.flow(x[ind], y_t[ind], batch_size=32):
        #            if turn * 32 > size-len(ind):
        #                break                    
        #            x_new.append(batch[0])
        #            y_new.append(batch[1])
        #            turn+=1
        #        print('sample_size:', len(np.concatenate(y_new,axis=0)))

        #    x_new = np.concatenate(x_new,axis=0)
        #    y_new = np.concatenate(y_new,axis=0)
        
        return x_new, y_new    

    def clone(self,key,newKey,active=True,description=None, replace_description=False,
             autosave=True):
        assert newKey not in self.keys()
        
        model = self.getModel(key)
        filetype = self.charts[key]['filetype']
        model2 = clone_model(model)
        filepath = self.modelPath(newKey,filetype)
        model2.save(filepath)
        self.charts[newKey] = {}
        self.charts[newKey]['target'] = self.charts[key]['target']
        self.charts[newKey]['active'] = active
        self.charts[newKey]['fuzDom'] = None
        self.charts[newKey]['filetype'] = filetype
        if self.charts[key]['spec for loading model'] is not None and self.charts[key]['spec for loading model']['name'] != 'Keras-pretrained':
            self.charts[newKey]['spec for loading model'] = self.charts[key]['spec for loading model']
        else:
            self.charts[newKey]['spec for loading model'] = None
        text = ""
        if not replace_description:
            text += "Clone from Model %s.\n"%(self.modelName(key))
        if description is not None:
            text += description + "\n"
        if not replace_description and self.charts[key]['description'] is not None:           
            text += "Description of Model %s: %s\n"%(self.modelName(key), 
                                                   self.charts[key]['description'])
        self.charts[newKey]['description'] = text
        if autosave:
            self.save()
        return model2                
    
    #Scale up softmax to make it more sensitive.
    #We save the scale to "scale_in" attribute which is needed for loading back model
    def scale_in(self,key,newKey,scale, active=True,keep_memory=True,
                 description=None,replace_description=False, autosave=True):
        assert newKey not in self.keys()
        
        model = self.getModel(key)
        filetype = self.charts[key]['filetype']
        assert len(model.output_shape)==2
        out = model.output_shape[1]
        assert isinstance(model.layers[-1],Dense)
        base = Model(inputs = model.inputs, outputs = model.layers[-2].output) 
        
        inputs = keras.Input(shape=base.layers[0].input_shape[0][1:])
        y = base(inputs)
        outputs = Dense(out,activation=softmax_scaled(scale))(y)
        model2 = Model(inputs=inputs, outputs=outputs)
        
        if keep_memory:
            model2.layers[-1].set_weights(model.layers[-1].get_weights())  
            
        filepath = self.modelPath(newKey,filetype)
        model2.save(filepath)
        self.charts[newKey] = {}
        self.charts[newKey]['target'] = self.charts[key]['target']
        self.charts[newKey]['active'] = active
        self.charts[newKey]['filetype'] = filetype
        self.charts[newKey]['spec for loading model'] = {'name':'scale_in', 'scale':scale}
        if keep_memory:
            self.charts[newKey]['fuzDom'] = self.charts[key]['fuzDom']
        else:
            self.charts[newKey]['fuzDom'] = None
        text = ""
        if not replace_description:
            text += "Scale-in of Model %s.\n"%(self.modelName(key))
            text += "Scale-in rate is %f"%scale
        if description is not None:
            text += description + "\n"
        if not replace_description and self.charts[key]['description'] is not None:           
            text += "Description of Model %s: %s\n"%(self.modelName(key), 
                                                   self.charts[key]['description'])
        self.charts[newKey]['description'] = text  
        if autosave:
            self.save()        
        return model2        

    #Increase the size of the second last dense layer
    #Assume the dense layer has equipped with activation
    def grow(self,key,newKey,growBy,active=True,keep_memory=True,
             description=None, replace_description=False, autosave=True):
        assert newKey not in self.keys()
        
        model = self.getModel(key)
        filetype = self.charts[key]['filetype']
        assert len(model.output_shape)==2
        out = model.output_shape[1]        
        denses = getDense(model)
        assert len(denses)>0
        
        if len(denses)==1: #Then add one more dense layer right before this one
            #cannot keep memory (for the modified layers) in this case
            #Assume dense is the last layer
            l = getDense(model)[0]
            base = Model(inputs = model.inputs, outputs = model.layers[l-1].output) 
            
            inputs = keras.Input(shape=base.layers[0].input_shape[0][1:])
            y = base(inputs)
            y = Dense(growBy(out),activation="relu")(y)
            outputs = Dense(out,activation="softmax")(y)
            model2 = Model(inputs=inputs, outputs=outputs)
            keep_memory=False
            
        else: #otherwise, increase the width of the second last dense layer
            l,m = getDense(model)[-2:]
            width = model.layers[l].output_shape[1]
            #base = Model(inputs = model.inputs, outputs = model.layers[l-1].output)
            act = model.layers[l].activation
            
            inputs = keras.Input(shape=model.layers[0].input_shape[0][1:])
            y = inputs
            for layer in model.layers[1:l]:
                y = layer(y)                           
            #inputs = keras.Input(shape=base.layers[0].input_shape[0][1:])
            #y = base(inputs)
            y = Dense(growBy(width),activation=act)(y)
            outputs = Dense(out,activation="softmax")(y)
            model2 = Model(inputs=inputs, outputs=outputs) 
            if keep_memory:
                oldW = model.layers[l].get_weights()
                newW = model2.layers[-2].get_weights()
                newW[0][:,:width] = oldW[0]
                newW[0][:,width:] = 0.
                newW[1][:width] = oldW[1]
                newW[1][width:] = 0.
                model2.layers[-2].set_weights(newW)
 
                oldW = model.layers[m].get_weights()
                newW = model2.layers[-1].get_weights()
                newW[0][:width,:] = oldW[0]
                newW[0][width:,:] = 0.
                newW[1] = oldW[1]
                model2.layers[-1].set_weights(newW)
  
        filepath = self.modelPath(newKey,filetype)
        model2.save(filepath)
        self.charts[newKey] = {}
        self.charts[newKey]['target'] = self.charts[key]['target']
        self.charts[newKey]['active'] = active
        self.charts[newKey]['filetype'] = filetype
        if self.charts[key]['spec for loading model'] is not None and self.charts[key]['spec for loading model']['name'] != 'Keras-pretrained':
            self.charts[newKey]['spec for loading model'] = self.charts[key]['spec for loading model']
        else:
            self.charts[newKey]['spec for loading model'] = None
        if keep_memory:
            self.charts[newKey]['fuzDom'] = self.charts[key]['fuzDom']
        else:
            self.charts[newKey]['fuzDom'] = None
        text = ""
        if not replace_description:
            text += "Grown up of Model %s.\n"%(self.modelName(key))
        if description is not None:
            text += description + "\n"
        if not replace_description and self.charts[key]['description'] is not None:           
            text += "Description of Model %s: %s\n"%(self.modelName(key), 
                                                   self.charts[key]['description'])
        self.charts[newKey]['description'] = text    
        if autosave:
            self.save()        
        return model2
        
    def turnSpecialist(self,key:tuple,newKey:tuple,newTarget,train:bool=True,epochs:int=30,
                       lr:float=1e-3,batch_size:int=32,lr_schedule=None, 
                       datagen=None, resample:bool=False,
                       iniTh:float=0.,step:float=0.5,minValPercentage:float=0.01,
                       active:bool=True, save_each:bool=False, 
                       saveAboveAcc : float =None, saveBests: int =None,description : str =None,
                       autosave:bool=True, verbose:int = 0):  
        assert newKey not in self.keys()
        newout = len(newTarget)
        
        model = self.getModel(key)
        filetype = self.charts[key]['filetype']
        assert len(model.output_shape)==2
        for i in range(len(model.weights)):
            if self.modelName(key) not in model.weights[i].name:
                model.weights[i]._handle_name = model.weights[i].name + "_" + self.modelName(key)
        for l in model.layers:
            if self.modelName(key) not in l.name:
                l._name = l.name + "_" + self.modelName(key)         
        
        #Change target of model and add to atlas
        #Take up to the layer before softmax.
        base = Model(inputs = model.inputs, outputs = model.layers[-2].output, name=self.modelName(newKey)+'base')             
        inputs = keras.Input(shape=model.layers[0].input_shape[0][1:],name=self.modelName(newKey)+'Input')
        y = base(inputs)
        y = Dense(newout,name=self.modelName(newKey)+'Dense')(y)
        outputs = layers.Softmax(name=self.modelName(newKey)+'Softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
        filepath = self.modelPath(newKey,filetype)
        model.save(filepath)
        
        self.charts[newKey] = {}
        self.charts[newKey]['fuzDom'] = None
        self.charts[newKey]['target'] = newTarget
        self.charts[newKey]['active'] = active
        self.charts[newKey]['filetype'] = filetype
        text = "Specialist coming from %s.\n"%(self.modelName(key))
        if description is not None:
            text += description + "\n"
        if self.charts[key]['description'] is not None:           
            text += "Description of %s: %s\n"%(self.modelName(key), 
                                                   self.charts[key]['description'])
        self.charts[newKey]['description'] = text
        #Since Keras-pretrained model saves like usual model, no spec is needed for loading the saved one.
        if self.charts[key]['spec for loading model'] is not None and self.charts[key]['spec for loading model']['name'] != 'Keras-pretrained':
            self.charts[newKey]['spec for loading model'] = self.charts[key]['spec for loading model']
        else:
            self.charts[newKey]['spec for loading model'] = None
        storyFile = self.storyFile
        #Train if there is training data
        already_written = False
        if train:
            assert self.x_tr is not None and self.y_tr is not None
            train_summary = self.train(newKey,epochs=epochs,lr=lr,batch_size=batch_size,
                       datagen=datagen, lr_schedule=lr_schedule, resample=resample,
                       iniTh=iniTh, step=step, minValPercentage=minValPercentage,
                       save_each=save_each, saveAboveAcc=saveAboveAcc, saveBests=saveBests, verbose= verbose)
            
            if storyFile is not None:
                with open(self.path+storyFile+".txt", "a") as story:
                    story.write("---TurnSpecialist---\n")
                    story.write(f"Model {key}\nmodel has specialized to {newTarget} and saved as {newKey}.\n{train_summary}\nCurrently our society has {len(self.charts.keys())} number of models.\n")
                already_written = True
        else:
            self.charts[newKey]['fuzDom'] = None
        
        if autosave:
            self.save()    
        
        if storyFile is not None and not already_written:
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---TurnSpecialist---\n")
                story.write(f"From model {key}, a model {newKey} specialized to {newTarget} has trained.\nCurrently our society has {len(self.charts.keys())} number of models.\n")            
        return   

    #Can make `organs' and combine them.
    #For instance, train a model with target [[0,1,2,3,4],[5,6,7,8,9]].
    #Then train two models, one with target [0,1,2,3,4] and one with target [5,6,7,8,9]
    #with subsamples.
    #Then combine them to a model.
    #The model with target [[0,1,2,3,4],[5,6,7,8,9]] can also be treated as
    #a single chart to vote.
    def marryBorn(self,keys,newKey,target=None,epochs : int =50,lr : float =1e-4,batch_size : int =32,
                      iniTh : float =0.,step : float =0.5,minValPercentage : float =0.01,datagen=None,
                      train : bool =True,active : bool =True,filetype : str =None,save_each : bool =False,
                      saveAboveAcc=None,lr_schedule=None,description : str =None,
                      spec=None,autosave : bool =True, verbose : int = 1):
        assert newKey not in self.keys()
        
        if target is None:
            target = [[i] for i in range(self.target)]
        out = len(target)
        bases = []
        
        for key in keys:
            model = self.getModel(key)
            for i in range(len(model.weights)):
                if self.modelName(key) not in model.weights[i].name:
                    model.weights[i]._handle_name = model.weights[i].name + "_" + self.modelName(key)
            for l in model.layers:
                if self.modelName(key) not in l.name:
                    l._name = l.name + "_" + self.modelName(key)
            last = getDense(model)[-1]
            bases.append(Model(inputs = model.inputs, outputs = model.layers[last].output, name=self.modelName(key)))
        
        if filetype is None:
            filetype = self.charts[key]['filetype']
        
        input_shape = model.layers[0].input_shape[0][1:]       
        inputs = keras.Input(shape=input_shape,name=self.modelName(newKey)+'Input')
        y = Concatenate(name=self.modelName(newKey)+'Concatenate')([bases[i](inputs) for i in range(len(keys))])
        y = Dense(out,use_bias=False,name=self.modelName(newKey)+'Dense')(y)
        outputs = layers.Softmax(name=self.modelName(newKey)+'Softmax')(y)
        model2 = Model(inputs=inputs, outputs=outputs)
        filepath = self.modelPath(newKey,filetype)        
        model2.save(filepath)
        
        self.charts[newKey] = {}
        self.charts[newKey]['fuzDom'] = None
        self.charts[newKey]['target'] = target
        self.charts[newKey]['active'] = active
        self.charts[newKey]['filetype'] = filetype
        self.charts[newKey]['spec for loading model'] = spec
        text = "Born by Models "
        for key in keys[:-1]:
            text += self.modelName(key) + ", "
        text += self.modelName(keys[-1]) + ".\n"
        if description is not None:
            text += description + "\n"
        self.charts[newKey]['description'] = text 

        #Train if there is training data
        if train and self.x_tr is not None and self.y_tr is not None:            
            train_summary=self.train(newKey,epochs=epochs,lr=lr,
                               batch_size=batch_size,datagen=datagen,
                               save_each=save_each, saveAboveAcc=saveAboveAcc,
                               lr_schedule=lr_schedule, verbose = verbose)
        else:   
            self.getFuzDoms(iniTh=iniTh, step=step, 
                       minValPercentage=minValPercentage, keys=[newKey])
               
        if autosave:
            self.save()        
        storyFile = self.storyFile
        if storyFile is not None:
            with open(self.path+storyFile+".txt", "a") as story:
                story.write("---MarryBorn---\n")
                story.write(f'{keys} give birth to \n{newKey}.\n{train_summary}\nCurrently our society has {len(self.charts.keys())} number of models.\n')            
        return     
    
    def train(self,key: tuple,x_tr=None,y_tr=None,x_v=None,y_v=None,
              new_key : tuple =None,active : bool =True,epochs : int =50,lr : float =1e-3,batch_size : int =32,
              datagen=None, resample : bool =False, emphasizes: list = [], emph_wt : float=1.,
              iniTh : float =0.,step : float =0.5,minValPercentage : float =0.01,lr_schedule=None,
              save_each : bool =False, newLabel : bool =True, monitor_coverage_level=None,
              saveAboveAcc : float =None,saveBests : int =None,verbose : int =0,description : str =None,
              autosave:bool=True):
        save_to_new = False
        if new_key is not None:
            assert new_key not in self.keys()
            save_to_new = True
        if x_tr is None:
            x_tr=self.x_tr
        if y_tr is None:
            y_tr=self.y_tr
        if x_v is None:
            x_v=self.x_v
        if y_v is None:
            y_v=self.y_v
        initial_description = str()  # Used when save_each = True for description of models at each epoch.
        model = self.getModel(key)
        filetype = self.charts[key]['filetype']
        opt = tf.keras.optimizers.Adam(learning_rate=lr) 
        
        if monitor_coverage_level is None:
             
            model.compile(optimizer=opt, 
                          loss=tf.keras.losses.CategoricalCrossentropy(), 
                          metrics=["accuracy"])  
                  
        else:
            covermet = coverage(monitor_coverage_level)
            model.compile(optimizer=opt, 
                          loss=tf.keras.losses.CategoricalCrossentropy(), 
                          metrics=["accuracy",covermet]) 
        target = self.charts[key]['target']       
        
        if resample:
            x_tr,y_tr = self.samplesForTarget(target,x_tr,y_tr, newLabel=newLabel)
            x_v,y_v = self.samplesForTarget(target,x_v,y_v, newLabel=newLabel)
        elif len(target)<self.target and newLabel:
            x_tr, y_tr = self.newLabelForTarget(x_tr, y_tr, target)
            x_v, y_v = self.newLabelForTarget(x_v, y_v, target)                
        if len(emphasizes) > 0:
            x_tr,y_tr = self.incWtsForTarget(x_tr,y_tr,emphasizes,wt=emph_wt)
            x_v,y_v = self.incWtsForTarget(x_v,y_v,emphasizes,wt=emph_wt) 
        if new_key is not None:
            self.charts[new_key]={}
            self.charts[new_key]['fuzDom'] = None
            self.charts[new_key]['target'] = self.charts[key]['target']
            self.charts[new_key]['active'] = active
            self.charts[new_key]['filetype'] = filetype
            
            #Since Keras-pretrained model saves like usual model, no spec is needed for loading the saved one.
            if self.charts[key]['spec for loading model'] is not None and self.charts[key]['spec for loading model']['name'] != 'Keras-pretrained':
                self.charts[new_key]['spec for loading model'] = self.charts[key]['spec for loading model']
            else:
                self.charts[new_key]['spec for loading model'] = None
            text = "Trained from Model %s.\n"%(self.modelName(key))
            if description is not None:
                text += description + "\n"
                initial_description += description + "\n"
            if self.charts[key]['description'] is not None:           
                text += "Description of Model %s: %s\n"%(self.modelName(key), 
                                                       self.charts[key]['description'])
                initial_description += "Description of Model %s: %s\n"%(self.modelName(key), 
                                                       self.charts[key]['description'])
            self.charts[new_key]['description'] = text
            if autosave:
                self.save()
        else:
            new_key = key
        if save_each:
            filepath = self.path + self.modelName(new_key) + '_{epoch:03d}.' + filetype
            checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy',
                                         verbose=verbose,save_best_only=False,save_weights_only=False)
        else:
            filepath = self.modelPath(new_key,filetype)
            if monitor_coverage_level is None:
                checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy',
                                             verbose=verbose,save_best_only=True,save_weights_only=False)
            else:
                checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_covermet',
                                             verbose=verbose,save_best_only=True,save_weights_only=False)
        
        if lr_schedule is not None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)            
            callbacks = [checkpoint, lr_reducer, lr_scheduler]
        else:
            callbacks = [checkpoint]
        
        if datagen is None:
            # If you put verbose = 1, then you can see training history output, which could induce delay.
            hist = model.fit(x_tr, y_tr, batch_size=batch_size, 
                      validation_data=(x_v, y_v), 
                      epochs=epochs, callbacks=callbacks, verbose=verbose)
        else:
            print('Using data generator...')
            datagen.fit(x_tr)
            steps_per_epoch =  math.ceil(len(x_tr) / batch_size)             
            hist = model.fit(x=datagen.flow(x_tr, y_tr, batch_size=batch_size),
                 validation_data=(x_v, y_v),
                 steps_per_epoch=steps_per_epoch, epochs=epochs, 
                             callbacks=callbacks, verbose=verbose)   
        
        train_summary = str()
        if save_each:
            target = self.charts[key]['target']
            for i in range(1,epochs+1):
                self.charts[new_key+(i,)]={}
                self.charts[new_key+(i,)]['fuzDom'] = None
                self.charts[new_key+(i,)]['target'] = target
                self.charts[new_key+(i,)]['active'] = active 
                self.charts[new_key+(i,)]['filetype'] = filetype
                #Since Keras-pretrained model saves like usual model, no spec is needed for loading the saved one.
                if self.charts[key]['spec for loading model'] is not None and self.charts[key]['spec for loading model']['name'] != 'Keras-pretrained':
                    self.charts[new_key+(i,)]['spec for loading model'] = self.charts[key]['spec for loading model']
                else:
                    self.charts[new_key+(i,)]['spec for loading model'] = None
                self.charts[new_key+(i,)]['description'] = initial_description + f'Specialized from {key}, saved during training {new_key} at {i}th epoch.'     
            keys=[new_key+(i,) for i in range(1,epochs+1)]
            if os.path.isfile(self.modelPath(new_key,filetype)):
                keys.append(new_key)
            else:
                self.charts.pop(new_key)
            self.getFuzDoms(iniTh=iniTh, step=step, 
                       minValPercentage=minValPercentage, keys=keys, autosave=autosave)
            removed_keys=[]
            if saveBests is not None:
                keys = self.orderByAcc(keys,flatTarget=flatTuple(target))
                removed_keys.extend(self.remove(keys[saveBests:],write_story=False))            
            if saveAboveAcc is not None:
                removed_keys.extend(self.remove(keys,accBelow=saveAboveAcc,write_story=False))
            keys_created = list(set(keys)-set(removed_keys))
            if not save_to_new:
                keys_created = list(set(keys_created)-set([new_key]))
            train_summary += f'{len(keys_created)} new keys created: {keys_created}\n'
        else:
            self.getFuzDoms(iniTh=iniTh, step=step, 
                       minValPercentage=minValPercentage, keys=[new_key], autosave=autosave)
        
        if monitor_coverage_level is None:        
            print("compared to %.3f in the first epoch."%(hist.history['val_accuracy'][0]))
            print("The best validation accuracy was %.3f"%(np.max(hist.history['val_accuracy'])))
            train_summary +=  "The best validation accuracy was %.3f"%(np.max(hist.history['val_accuracy']))+" compared to %.3f in the first epoch."%(hist.history['val_accuracy'][0])
        else:
            print("compared to %.3f in the first epoch."%(hist.history['val_covermet'][0]))   
            print("The best validation coverage was %.3f"%(np.max(hist.history['val_covermet'])))
            train_summary += "The best validation coverage was %.3f"%(np.max(hist.history['val_covermet']))+" compared to %.3f in the first epoch."%(hist.history['val_covermet'][0]) 
        return train_summary
    
    def adversarial(self, key : tuple, new_key : tuple, ad_key=None,
               epochs:int =15,lr:float=1e-4,batch_size:int=32,
               iniTh:float=0.,step:float=0.5,minValPercentage:float=0.01,description:str =None,autosave:bool=False):
        
        assert new_key not in self.keys()
        
        if ad_key is None:
            ad_key = key
        
        assert self.dataType == 'numpy'
        #For dataset or datagen, we can make new datagen that will filter away the `good' data on the fly.
        #But this is a bit more complicated and we will leave it to future versions.
        
        x_tr = self.x_tr
        y_tr = self.y_tr
        x_v = self.x_v  
        y_v = self.y_v
        
        target = self.charts[key]['target']
        assert target == self.charts[ad_key]['target']
        x_tr, y_tr = self.newLabelForTarget(x_tr, y_tr, target)
        x_v, y_v = self.newLabelForTarget(x_v, y_v, target)         
        
        model = self.getModel(ad_key)
        x_tr, y_tr = next(generate_adversarial_batch(model, len(x_tr),x_tr, y_tr, model.input_shape[1:], eps=0.1))
        x_v, y_v = next(generate_adversarial_batch(model, len(x_v),x_v, y_v, model.input_shape[1:], eps=0.1))
        text = "Trained to defend adversarial attack on Model %s.\n"%(self.modelName(ad_key))
        text = "Parent is Model %s.\n"%(self.modelName(key))
        if description is not None:
            text += description + "\n"
        if self.charts[key]['description'] is not None:           
            text += "Description of Model %s: %s\n"%(self.modelName(ad_key), 
                                                   self.charts[key]['description'])
            text += "Description of parent Model %s: %s"%(self.modelName(key), 
                                                   self.charts[key]['description'])            
        self.train(key,new_key=new_key, x_tr=x_tr,y_tr=y_tr,x_v=x_v,y_v=y_v,
              epochs=epochs,lr=lr,batch_size=batch_size,
              iniTh=iniTh,step=step,minValPercentage=minValPercentage,
              newLabel=False, description=text)
        if autosave:
            self.save()        
        return
    
    #The idea of `migrate' is:
    #form a subset of validation data that were uncertain or predicted wrong,
    #and use this as validation data to select the best model.
    #Thus we are selecting model that behave the best 
    #in those validation data that were not covered previously.
    #Thus we should not "save_each" during training, or otherwise we lose the meaning
    #of selection.
    #We also choose wrongly predicted or uncertain data by a reference model to 
    #form the training data.
    #This reference model can be random (if rand_key is True),
    #or just taken to be the same as the model we are migrating.    
    def migrate(self,key : tuple,new_key: tuple,wantAcc:float=0.,
               epochs:int=30,lr:float=1e-4,batch_size:int=128, training_ratio:float=0.8,
               iniTh:float=0.,step:float=0.5,minValPercentage:float=0.01,
               rand_key:bool=False, datagen=None,findValDom:bool=True,
               lr_schedule=None,save_each:bool=False, resample:bool=False,
               saveAboveAcc=None, saveBests=None,
               monitor_coverage_level=None, growBy=None, scale_in=None, reset:bool=False,
               verbose:int=0,description : str =None, autosave:bool=True): 
        assert new_key not in self.keys()
        assert key in self.keys()
        
        assert self.dataType == 'numpy'
        #For dataset or datagen, we can make new datagen that will filter away the `good' data on the fly.
        #But this is a bit more complicated and we will leave it to future versions.
        
        x_tr = self.x_tr
        y_tr = self.y_tr
        x_v = self.x_v  
        y_v = self.y_v
        
        if self.valDom is not None and not findValDom:
            bad = ~self.valDom
        else:
            bad = ~self.findValDom(wantAcc,verbose=verbose)        
        x_v = x_v[bad]
        y_v = y_v[bad]
        print("Size of prepared validation data: %d"%len(y_v))
        
        #Change to model target
        target = self.charts[key]['target']
        x_tr, y_tr = self.newLabelForTarget(x_tr, y_tr, target)
        x_v, y_v = self.newLabelForTarget(x_v, y_v, target) 
        
        #Do the same for training data.  
        #But just consider those predicted wrong or uncertain in this model.
        #Otherwise take too much time to predict using all the models.
        
        if rand_key:
            keys = self.keysInTarget(target)
            r = np.random.choice(range(len(keys)))
            ref = keys[r]
        else:
            ref = key
            
        # fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist : output of predict.
        predictions_result = self.predict(self.x_tr, wantAcc, keys=[ref],
                                          active=False, verbose=verbose)
        fullAns, certPart = predictions_result[0:2]
        #Since self.predict always pretend wantAcc is a list and returns a list, we need to take [0] here
        modelAns = np.argmax(fullAns[0],axis=-1) 
        
        correct = np.argmax(y_tr,axis=-1)==modelAns
        if wantAcc != 0.:
            index = 1
        else:
            index = 0 
        certain = certPart[index] #recall that certPart is a mask consisting of True/False
        
        good = ((correct)*(certain)==1)     
        ind = np.arange(len(y_tr))[good]
        ind = np.random.choice(ind, size=max(0,int(len(y_tr) * training_ratio) 
                                             - np.sum(~good)))
        if len(ind)>0:
            x_tr = np.concatenate([x_tr[ind], x_tr[~good]],axis=0)        
            y_tr = np.concatenate([y_tr[ind], y_tr[~good]],axis=0) 
        else:
            x_tr = x_tr[~good]
            y_tr = y_tr[~good]
        print("Size of training data:%d"%(len(y_tr)))
        print("Among them, %d were predicted incorrect and %d were uncertain."
              %(np.sum(~correct),np.sum(~certain)))
        text = "Migrated from Model %s.\n"%(self.modelName(key))
        if rand_key:
            text = "Trained on uncertain or wrongly predicted samples of Model %s.\n"%(self.modelName(ref))
        if description is not None:
            text += description + "\n"
        if self.charts[key]['description'] is not None:           
            text += "Description of Model %s: %s\n"%(self.modelName(key), 
                                                   self.charts[key]['description'])
                    
        if growBy is None:
            if scale_in is None:            
                #since we have applied newLabelForTarget already,
                #we cannot apply again in the training function.
                if reset:
                    text += "Have reset weights.\n"
                    self.clone(key,new_key,description=text, replace_description=True)
                    train_summary = self.train(new_key, x_tr=x_tr,y_tr=y_tr,x_v=x_v,y_v=y_v,
                          epochs=epochs,lr=lr,batch_size=batch_size,datagen=datagen,
                          iniTh=iniTh,step=step,minValPercentage=minValPercentage,
                          lr_schedule=lr_schedule,resample=resample,
                          newLabel=False, monitor_coverage_level=monitor_coverage_level,
                          save_each=save_each,saveAboveAcc=saveAboveAcc, saveBests=saveBests,
                          verbose=verbose)                    
                else:
                    train_summary = self.train(key,new_key=new_key, x_tr=x_tr,y_tr=y_tr,x_v=x_v,y_v=y_v,
                          epochs=epochs,lr=lr,batch_size=batch_size,datagen=datagen,
                          iniTh=iniTh,step=step,minValPercentage=minValPercentage,
                          lr_schedule=lr_schedule,resample=resample,
                          newLabel=False, monitor_coverage_level=monitor_coverage_level,
                          save_each=save_each,saveAboveAcc=saveAboveAcc, saveBests=saveBests,
                          verbose=verbose)
            else:
                text += "Scaled in by %f\n"%scale_in
                self.scale_in(key,new_key,scale_in, description=text, replace_description=True)
                train_summary = self.train(new_key, x_tr=x_tr,y_tr=y_tr,x_v=x_v,y_v=y_v,
                      epochs=epochs,lr=lr,batch_size=batch_size,datagen=datagen,
                      iniTh=iniTh,step=step,minValPercentage=minValPercentage,
                      lr_schedule=lr_schedule,resample=resample,
                      newLabel=False, monitor_coverage_level=monitor_coverage_level,
                      save_each=save_each,saveAboveAcc=saveAboveAcc, saveBests=saveBests,
                      verbose=verbose)                 
        else:
            text += "Grown up by %f\n"%growBy
            self.grow(key,new_key,growBy, description=text, replace_description=True)
            train_summary = self.train(new_key, x_tr=x_tr,y_tr=y_tr,x_v=x_v,y_v=y_v,
                  epochs=epochs,lr=lr,batch_size=batch_size,datagen=datagen,
                  iniTh=iniTh,step=step,minValPercentage=minValPercentage,
                  lr_schedule=lr_schedule,resample=resample,
                  newLabel=False, monitor_coverage_level=monitor_coverage_level,
                  save_each=save_each,saveAboveAcc=saveAboveAcc, saveBests=saveBests,
                  verbose=verbose) 
        if autosave:
            self.save() 
        self.charts[new_key]['description'] = text 
        storyFile = self.storyFile
        if storyFile is not None:
            with open(self.path+storyFile+".txt", "a") as story:
                story.write('---Migration---\n')
                story.write(f'{key} migrated to {new_key}, trained on {len(y_v)} numbers of validation data produced by Model {ref}.\n')
                story.write(f'{train_summary}\nCurrently our society has {len(self.charts.keys())} number of models.\n')                
        return

    def randKey(self, active=True):
        keys = list(self.keys(active=active))        
        r = np.random.choice(range(len(keys)))
        return keys[r]
    
    def makeTargetTree(self, keys=None, active=True):
        if keys is None:
            keys = list(self.charts.keys())       
        groups = self.groupByFlatTarget(keys=keys,active=active)
        targetTuples = list(groups.keys())
        targetTuples.sort(key = len,reverse=True)
        numNextNodes = {}
        targetTree = {}
        keysTree = {}
        
        numNextNodes[()] = 0
        targetTree[()] = tuple(range(self.target))
        keysTree[()] = groups[tuple(range(self.target))]
        targetTuples.remove(tuple(range(self.target)))
        '''
        2024 Oct 15.
        In old-version:
        
        Given {0,1,2,3}, {0,1,2}, {1,2,3}, {1,2}, the tree is following:
        {():(0,1,2,3), (0,):(0,1,2), (1,): (1,2,3), (0,0):(1,2)}
        In graph, it is following:
                    {0,1,2,3}
                {0,1,2}     {1,2,3}
            {1,2}
        But we want to have 
                    {0,1,2,3}
                {0,1,2}     {1,2,3}
            {1,2}               {1,2}
        That is
        {():(0,1,2,3), (0,):(0,1,2), (1,): (1,2,3), (0,0):(1,2), (1,0):(1,2)}.
        
        In old version, it breaks the for loop when it find a longest path from the root to the current node.
        But there could be other path, therefore I have removed the break. Then tree is going to be wild as any node can be attached under the root.
        Therefore I put recording code snippet of past-node, since each node is of the tuple form and it records its path. For instance, node (0,0,1,0) is following ()->(0,)->(0,0)->(0,0,1)->(0,0,1,0) (curr)
        
        So, given a node, if parent node == recoreded node sliced by the length of parent node, for instance parent node = (0,0) and we already put (0,0,1,0) having sliced to (0,0),
        we skip this parent node as it has (0,0,1) node between them.
        If parent node was (0,1), then we place (0,1,*) as a new child of (0,1).
         
        '''
        for targetTuple in targetTuples:
            keys = list(targetTree.keys())
            keys.sort(key = len,reverse=True)
            recording_of_assigned_nodes = []          
            for node in keys:
                if set(targetTuple).issubset(set(targetTree[node])) and set(targetTuple) != set(targetTree[node]):
                    if not recording_of_assigned_nodes:
                        # If there was no recording and we find child-parent relation
                        recording_of_assigned_nodes.append(node+(numNextNodes[node],))
                        targetTree[node+(numNextNodes[node],)] = targetTuple
                        numNextNodes[node+(numNextNodes[node],)] = 0
                        keysTree[node+(numNextNodes[node],)] = groups[targetTuple]
                        numNextNodes[node] +=1    
                    else:                
                        indicator = True
                        for past_node in recording_of_assigned_nodes:
                            if node == past_node[:len(node)]:
                                indicator = False
                                break
                        if indicator:
                            recording_of_assigned_nodes.append(node+(numNextNodes[node],))
                            targetTree[node+(numNextNodes[node],)] = targetTuple
                            numNextNodes[node+(numNextNodes[node],)] = 0
                            keysTree[node+(numNextNodes[node],)] = groups[targetTuple]
                            numNextNodes[node] +=1                            
        return targetTree, keysTree
        
    def groupByFlatTarget(self, keys=None, active=True):
        groups = {}
        if keys is None:
            keys = self.charts.keys()
        for k in keys:
            if active and not self.charts[k]['active']:
                continue
            target = self.charts[k]['target']
            # Given target could be grouped. Take (ordered) union of target groups.
            targetFlatTuple = flatTuple(target)
            # Still it may not be full. Make an hash table whose key is targets.
            if targetFlatTuple not in groups:
                groups[targetFlatTuple] = []        
            groups[targetFlatTuple].append(k)
        return groups
    
    def showTarget(self, keys = None, active=True):
        if keys is None:
            keys = self.charts.keys()
        for k in keys:
            if active and not self.charts[k]['active']:
                continue
        groups = self.groupByTarget(keys = keys, active=active)
        out = {k:len(groups[k]) for k in groups}
        return pd.DataFrame.from_dict(out,orient='index')
    
    def addCommittee(self, keys, name):
        self.committees[name] = keys
        return
    
    def showDescription(self, keys=None, active=False):
        if keys is None:
            keys = self.keys(active=active)
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            print("Model %s:"%self.modelName(k))
            print(self.charts[k]['description'])
    
    def showStory(self):
        with open(self.path+self.storyFile+".txt", "r") as story:
            print(story.read())
        return
    
    def save(self,save_data=False,update=True):
        assert self.path != None
        domain_targets={}
        for k in self.charts.keys():
            domain_targets[k] = (self.charts[k]['fuzDom'],
                                 self.charts[k]['target'],
                                 self.charts[k]['active'],
                                 self.charts[k]['description'],
                                 self.charts[k]['spec for loading model'])

        with open(self.path+'domain_targets.pkl', 'wb') as f:
            pickle.dump(domain_targets,f)
        
        if self.valDom is not None:
            with open(self.path+'valDom.npy', 'wb') as f:
                np.save(f,self.valDom)               
        if self.contexts is not None:
            with open(self.path+'contexts.pkl', 'wb') as f:
                pickle.dump(self.contexts,f)    
        if save_data:
            if not update or not os.path.isfile(self.path+'x_tr.npy') or not os.path.isfile(self.path+'y_tr.npy'):
                if self.x_tr is not None and self.y_tr is not None and self.dataType=='numpy':
                    with open(self.path+'x_tr.npy', 'wb') as f:
                        np.save(f,self.x_tr)                    
                    with open(self.path+'y_tr.npy', 'wb') as f:
                        np.save(f,self.y_tr)
            if not update or not os.path.isfile(self.path+'x_v.npy') or not os.path.isfile(self.path+'y_v.npy'):                    
                if self.x_v is not None and self.y_v is not None and self.dataType=='numpy':
                    with open(self.path+'x_v.npy', 'wb') as f:
                        np.save(f,self.x_v)
                    with open(self.path+'y_v.npy', 'wb') as f:
                        np.save(f,self.y_v)                
        return
    
    def save_logifold(self,update:bool=True):
        self.save(save_data=True,update=update)
        info = {'path':self.path, 'target':self.target, 'name':self.name,
                'storyFile':self.storyFile, 'dataType':self.dataType,
                'ds_tr_info':self.ds_tr_info, 'ds_val_info':self.ds_val_info,
                'input_shape':self.input_shape}
        if not isinstance(self.ds_tr_info,dict):
            info['ds_tr_info']=None
            if self.dataType != 'numpy':
                print('ds_tr_info is not saved.  Please provide x_tr or ds_tr_info in use of load_logifold.')
        if not isinstance(self.ds_val_info,dict):
            info['ds_val_info']=None
            if self.dataType != 'numpy':
                print('ds_val_info is not saved.  Please provide x_v or ds_val_info in use of load_logifold.')
        if self.name is not None:
            with open(self.path+self.name+'_committees.pkl', 'wb') as f:
                        pickle.dump(self.committees,f)        
            with open(self.path+self.name+'_logifold.pkl', 'wb') as f:
                        pickle.dump(info,f)
        else:
            with open(self.path+'committees.pkl', 'wb') as f:
                        pickle.dump(self.committees,f)        
            with open(self.path+'logifold.pkl', 'wb') as f:
                        pickle.dump(info,f)            
        return
        
    def load(self):
        assert self.path != None
        if os.path.isfile(self.path+'domain_targets.pkl'):
            with open(self.path+'domain_targets.pkl', 'rb') as f:
                domain_targets=pickle.load(f)
        else:
            domain_targets={}
        
        if os.path.isfile(self.path+'valDom.npy'):
            with open(self.path+'valDom.npy', 'rb') as f:
                self.valDom=np.load(f)    
        if os.path.isfile(self.path+'contexts.pkl'):
            with open(self.path+'contexts.pkl', 'rb') as f:
                self.contexts=pickle.load(f)
        
        #We automatically load all model files, no matter whether
        #their corresponding keys were recorded in file or not.
        modelFiles = [f for f in os.listdir(self.path) if f.startswith("model")]
        for f in modelFiles:
            fsplit = f.split('_')
            key = ()
            for i in range(1,len(fsplit)-1):
                key+=(int(fsplit[i]),)
            end = fsplit[-1].split('.')
            key+=(int(end[0]),)
            filetype = end[1]
            self.charts[key] = {}
            
            if key in domain_targets:
                self.charts[key]['fuzDom'] = domain_targets[key][0]
                self.charts[key]['target'] = domain_targets[key][1]
                self.charts[key]['active'] = domain_targets[key][2]
                self.charts[key]['filetype'] = filetype                
                self.charts[key]['description'] = domain_targets[key][3]
                self.charts[key]['spec for loading model'] = domain_targets[key][4]
            else:
                self.charts[key]['fuzDom'] = None
                self.charts[key]['target'] = [[i] for i in range(self.target)]
                self.charts[key]['active'] = True
                self.charts[key]['filetype'] = filetype
                self.charts[key]['description'] = None
                self.charts[key]['spec for loading model'] = None
        if self.storyFile is not None:    
            story = open(self.path+self.storyFile+".txt","a")
            L = ["---Loading---\nOur society has imported %d models.\n"%len(modelFiles)]
            story.writelines(L)
            story.close() #to change file access modes                
        
        #if model is keras-pretrained, then just load domain_targets
        for k in domain_targets.keys():
            if isinstance(k[0],str):
                self.charts[k] = {}
                self.charts[k]['fuzDom'] = domain_targets[k][0]
                self.charts[k]['target'] = domain_targets[k][1]
                self.charts[key]['active'] = domain_targets[key][2]
                self.charts[key]['description'] = domain_targets[key][3]
                self.charts[key]['spec for loading model'] = domain_targets[key][4]
        return
        
def load_logifold(folder,name=None,ds_tr_info=None,ds_val_info=None, x_tr=None, x_v=None):
    if name is not None:
        with open(folder+name+'_logifold.pkl', 'rb') as f:
            info=pickle.load(f)
        with open(folder+name+'_committees.pkl', 'rb') as f:
            committees=pickle.load(f)
    else:
        with open(folder+'logifold.pkl', 'rb') as f:
            info=pickle.load(f)
        with open(folder+'committees.pkl', 'rb') as f:
            committees=pickle.load(f)
    
    if 'dataType' in info:
        dataType = info['dataType']
    else:
        dataType = 'numpy'
    if dataType == 'numpy':
        logi = Logifold(info['target'], name=name,
                        path=info['path'], load_val_path=True, load_tr_path=True, 
                        storyFile=info['storyFile'], new_story=False)
    elif dataType == 'ds' or dataType == 'datagen':
        if ds_tr_info is None:
            ds_tr_info = info['ds_tr_info']
        if ds_val_info is None:
            ds_val_info = info['ds_val_info'] 
        input_shape = info['input_shape']
        logi = Logifold(info['target'], name=name,
                        path=info['path'],
                        dataType=dataType, ds_tr_info=ds_tr_info, ds_val_info=ds_val_info, input_shape=input_shape,
                        x_tr=x_tr,x_v=x_v,
                        storyFile=info['storyFile'], new_story=False)        
    else:
        print('Data type not recognized.  No training nor validation data is loaded.')
        
    logi.committees = committees    
    if os.path.isfile(info['path']+'valDom.npy'):
        with open(info['path']+'valDom.npy', 'rb') as f:
            logi.valDom = np.load(f)
    if os.path.isfile(info['path']+'contexts.pkl'):
        with open(info['path']+'contexts.pkl', 'rb') as f:
            logi.contexts = pickle.load(f)            
            
    #record filetypes of model files
    modelFiles = [f for f in os.listdir(info['path']) if f.startswith("model")]
    filetypes = {}
    for f in modelFiles:
        fsplit = f.split('_')
        key = ()
        for i in range(1,len(fsplit)-1):
            key+=(int(fsplit[i]),)
        end = fsplit[-1].split('.')
        key+=(int(end[0]),)
        filetypes[key] = end[1]             
            
    with open(info['path']+'domain_targets.pkl', 'rb') as f:
        domain_targets=pickle.load(f)             
    for key in domain_targets:
        logi.charts[key] = {}
        logi.charts[key]['fuzDom'] = domain_targets[key][0]
        logi.charts[key]['target'] = domain_targets[key][1]
        logi.charts[key]['active'] = domain_targets[key][2]
        logi.charts[key]['description'] = str(domain_targets[key][3])
        logi.charts[key]['spec for loading model'] = domain_targets[key][4]
        logi.charts[key]['filetype'] = filetypes[key]       
    return logi
    
def flatTuple(target,ascending=True):
    out = ()
    for a in target:
        out+=tuple(a)
    if ascending:
        out = tuple(sorted(out))
    return out

#Turn [[4,2],[3]] to [[2,0],[1]]
def reIndex(target):
    flatten = flatTuple(target,ascending=True)    
    out = []
    for t in target:          
        out.append([flatten.index(i) for i in t])
    return out

def generate_image_adversary(model, image, label, eps=2 / 255.0):
	# cast the image
	image = tf.cast(image, tf.float32)
	# record our gradients
	with tf.GradientTape() as tape:
		# explicitly indicate that our image should be tacked for
		# gradient updates
		tape.watch(image)
		# use our model to make predictions on the input image and
		# then compute the loss
		pred = model(image)
		loss = MSE(label, pred)
	# calculate the gradients of loss with respect to the image, then
	# compute the sign of the gradient
	gradient = tape.gradient(loss, image)
	signedGrad = tf.sign(gradient)
	# construct the image adversary
	adversary = (image + (signedGrad * eps)).numpy()
	# return the image adversary to the calling function
	return adversary

def generate_adversarial_batch(model, total, images, labels, dims,
	eps=0.01):
	# unpack the image dimensions into convenience variables
	(h, w, c) = dims
	# we're constructing a data generator here so we need to loop
	# indefinitely
	while True:
		# initialize our perturbed images and labels
		perturbImages = []
		perturbLabels = []
		# randomly sample indexes (without replacement) from the
		# input data
		idxs = np.random.choice(range(0, len(images)), size=total,
			replace=False)
		# loop over the indexes
		for i in idxs:
			# grab the current image and label
			image = images[i]
			label = labels[i]
			# generate an adversarial image
			adversary = generate_image_adversary(model,
				image.reshape(1, h, w, c), label, eps=eps)
			# update our perturbed images and labels lists
			perturbImages.append(adversary.reshape(h, w, c))
			perturbLabels.append(label)
		# yield the perturbed images and labels
		yield (np.array(perturbImages), np.array(perturbLabels))
        
        

def coverage(k):
    def measure(y_true, y_pred):
        count_certain = K.sum(K.cast((K.max(y_pred,axis=-1) > sigmoid(float(k))),'int32'))
        return count_certain / len(y_pred)
    return measure        
    
def softmax_scaled(k):
    def softmax_k(x):
        return K.softmax(k * x)
    return softmax_k

def relu_scaled(k):
    def f(x):
        return K.relu(k * x)
    return f

def sigmoid_scaled(k):
    def f(x):
        return K.sigmoid(k * x)
    return f

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getDense(model):    
    return [i for i in range(len(model.layers)) if isinstance(model.layers[i],Dense)]

class StopOnceHit(tf.keras.callbacks.Callback):
    def __init__(self, acc):
        super().__init__()
        self.acc = acc
        
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= self.acc:
            self.model.stop_training = True
            
def resnet_v1(input_shape, depth, num_classes):
    """ResNet Version 1 Model builder [a]
    
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved
    (downsampled) by a convolutional layer with strides=2, while 
    the number of filters is doubled. Within each stage, 
    the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    
    Arguments:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    
    Returns:
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, in [a])')
    # start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                # linear projection residual shortcut
                # connection to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    
    # add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    
    # instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes):
    """ResNet Version 2 Model builder [b]
    
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or 
    also known as bottleneck layer.
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, 
    the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, 
    while the number of filter maps is
    doubled. Within each stage, the layers have 
    the same number filters and the same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    
    Arguments:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    
    Returns:
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 110 in [b])')
    # start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)
    
    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU
    # on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)
    
    # instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                # first layer and first stage
                if res_block == 0:  
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                # first layer but not first stage
                if res_block == 0:
                    # downsample
                    strides = 2 
    
            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection
                # to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
    
        num_filters_in = num_filters_out
    
    # add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    
    # instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model            

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    Arguments:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    Returns:
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x