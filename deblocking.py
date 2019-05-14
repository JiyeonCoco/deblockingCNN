
import os
import struct
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import random
import logging
import time


testSetPath = './Testset'
tempPath = './temp_jy'
modelPath = './model_jy/'
tensorboardLogPath = './tensorboard_jy/'

NumberK = 12
nb_between = 2
nb_addition_layer = 3

learningRate = 1e-4
epsilon = 1e-4  # AdamOptimizer epsilon

os.makedirs(modelPath, exist_ok=True)
os.makedirs(tensorboardLogPath, exist_ok=True)


class LoggingHelper(object):
    INSTANCE = None

    def __init__(self):
        if LoggingHelper.INSTANCE is not None:
            raise ValueError("An instantiation already exists!")

        os.makedirs(tempPath, exist_ok=True)
        self.logger = logging.getLogger()

        logging.basicConfig(filename='LOGGER', level=logging.DEBUG)

        fileHandler = logging.FileHandler(tempPath + '/msg.log')
        streamHandler = logging.StreamHandler()

        fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler.setFormatter(fomatter)
        streamHandler.setFormatter(fomatter)

        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

    @classmethod
    def getInstace(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = LoggingHelper()
        return cls.INSTANCE

    @staticmethod
    def diff_time_logger(messege, start_time):
        LoggingHelper.getInstace().logger.info("[{}] :: running time {}".format(messege, time.time() - start_time))

logger = LoggingHelper.getInstace().logger




def getPsnr(loss):
    if loss <= 0:
        return 100
    return math.log10(1/loss) * 10


def concatenation(layers):
    return tf.concat(layers, axis=-1)


def conv2d(inputImg, numOutput, kernelSize, stride=1, padding='SAME', activation=tf.nn.relu, training=True, name='conv'):
    assert padding in ['SYMMETRIC', 'VALID', 'SAME', 'REFLECT']

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int((kernelSize - 1) / 2)
        inputImg = tf.pad(inputImg, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'

    inputImg = tf.layers.conv2d(inputImg, numOutput, kernelSize, stride, padding=padding, activation=activation, training=training, name=name)

    return inputImg


def batchActivConv(x, out_features, kernel_size, is_training, activation=None, rate=1, name="layer"):
    with tf.variable_scope(name):
          # no dropout
            x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
            x = tf.nn.relu(x)
            x = conv2d(x, out_features, kernel_size, stride=rate, activation=activation)

            return x


def relu(inputData):
    return tf.nn.relu(inputData)


def batchNormalization(inputData, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=inputData, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=inputData, is_training=training, reuse=True))


def bottleneckLayer(inputData, scope, training, filters):
    with tf.name_scope(scope):
        inputData = tf.contrib.layers.batch_norm(inputData, scale=True, is_training=training, updates_collections=None)
        inputData = relu(inputData)
        inputData = conv2d(inputData, 4 * filters, 1, name = scope + '_conv1')
        inputData = tf.contrib.layers.batch_norm(inputData, scale=True, is_training=training, updates_collections=None)
        inputData = relu(inputData)
        inputData = conv2d(inputData, filters, 3, name = scope + '_conv2')

        return inputData


def denseBlock(input_x, nb_layers, training, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneckLayer(input_x, scope=layer_name+'_bottleN_'+str(0), training = training, filters = NumberK)

        for i in range(nb_layers - 1):
            x = concatenation(layers_concat)
            x = bottleneckLayer(x, scope=layer_name + '_bottleN'+str(i+1), training = training, filters = NumberK)
            layers_concat.append(x)
        x = concatenation(layers_concat)

        return x


def mainNetwork(inputdata, training, nb_layers=5, reuse=False, name='Separable'):
    with tf.variable_scope(name + "_scope", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variable()
        conv1 = conv2d(inputdata, 64, 3, padding='SAME', name=name + 'conv1')
        conv2 = conv2d(conv1, 32, 3, padding='SAME', name=name + 'conv2')
        layers_concat = list()
        layers_concat.append(conv2)

        for i in range(nb_layers - 1):
            x = concatenation(layers_concat)
            x = bottleneckLayer(x, scope=name + '_bottleN' + str(i + 1), training=training, filters=NumberK)
            layers_concat.append(x)

        x = concatenation(layers_concat)
        layers_concat = list()
        x = tf.layers.conv2d(x, 32, [3, 3], padding='SAME', name=name + 'conv3')
        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = concatenation(layers_concat)
            x = bottleneckLayer(x, scope=name + '_bottleN2' + str(i + 1), training=training, filters=NumberK)
            layers_concat.append(x)

        x = concatenation(layers_concat)
        x = tf.layers.conv2d(x, 2, [3, 3], padding='SAME', name=name + 'conv4')

        return x

def resultNetwork(reuse = False, name = 'Common', training = True):
    with tf.variable_scope(name+"_scope", reuse = reuse) as scope:
        if reuse:
            scope.reuse_variable()

        orgData = tf.placeholder(tf.float32, shape = [None, None, None, 2], name="orgData")/1023.
        unfiltData = tf.placeholder(tf.float32, shape = [None, None, None, 8], name = "unfiltData")/1023.
        resData = tf.placeholder(tf.float32, shape = [None, None, None, 2], name = 'resData')/1023.

        result = mainNetwork(unfiltData, training, name = name + "Net", nb_layers=8)

        # orgdata = tf.slice(orgdata, [0, y, x, 1], [tf.shape(orgdata)[0], tf.subtract(h,y), tf.subtract(w, x), 2])
        result = tf.slice(result, [0, y, x, 0], [tf.shape(result)[0], 12, 12, 2])
        # preddata = tf.slice(preddata, [0, y, x, 1], [tf.shape(preddata)[0], tf.subtract(h,y), tf.subtract(w, x), 2])

        L1_loss = tf.reduce_mean(tf.losses.absolute_difference(labels = orgData, predictions= result))
        L2_loss = tf.losses.mean_squared_error(labels = orgData, predictions = result)
        CL2_loss = tf.losses.mean_squared_error(labels = orgData, predictions = resData)

        return L1_loss, L2_loss, CL2_loss


class Trainer(object):
    batchSize = 6
    trainingSetPath = "./Dataset/Training"

    def __init__(self):
        self.iter = 0

        self.trainingSetList = self.getTrainingDataList(Trainer.trainingSetPath)
        self.NumberOfTrainingShape = len(self.trainingSetList)
        self.initTrainingSet, self.dataSize = self.setTrainingData()
        self.dataOrder = np.arange(self.dataSize)
        #np.random.shuffle(self.dataOrder)
        logger.info("TrainingSet Build Finished : %d" %self.dataSize)
        self.L1_loss, self.L2_loss, self.CL2_Loss = resultNetwork(reuse=False, name='Common', training=True)
        self.optimizer =  tf.train.AdamOptimizer(learning_rate=learningRate, epsilon=epsilon)
        self.opt = self.optimizer.minimize(self.L1_loss)
        self.writer = tf.summary.FileWriter(tensorboardLogPath + '_CNN_Prediction')
        self.Cwriter = tf.summary.FileWriter(tensorboardLogPath + '_VVC_Prediction')
        self.log = tf.Variable(0.0)
        self.summary = tf.summary.scalar("Training Sequences", self.log)
        logger.info("Graph Build Finished")


    # 디렉토리 내에 있는 bin file 차례대로 읽어서 구성 (sequence)
    def getTrainingDataList(self, path, pattern='.bin'):
        sequences = []

        for root, dirNames, fileNames in os.walk(path):
            for fileName in fileNames:
                if fileName.endswith(pattern):
                    sequences.append(os.path.join(root, fileName))

        return sequences


    # training dataset을 batch size에 맞춰 구성 (dataset)
    def setTrainingData(self):
        dataset = { 'org' : [], 'unfilt' : [], 'res' : [] }

        for fileList in self.trainingSetList:

            if self.NumberOfTrainingShape % self.batchSize != 0:
                fileList = fileList[:-(self.NumberOfTrainingShape % self.batchSize)]
                fileList = np.array(fileList).reshape(-1, self.batchSize)

            for batchList in fileList:
                org = []
                unfilt = []
                res = []

                for idx, filePath in enumerate(batchList):
                    orgData, unfiltData = self.unpackTrainingData(idx, filePath)

                    org.append(orgData)
                    unfilt.append(unfiltData)
                    res.append(orgData[idx] - unfiltData[idx])

                dataset['org'].append(np.array(org))
                dataset['unfilt'].append(np.array(unfilt))
                dataset['res'].append(np.array(res))

        return dataset, len(dataset['res'])


    # bin file로부터 data 읽어서 각각 구성 (original, unfiltered)
    def unpackTrainingData(self, idx, imgPath):
        with open(imgPath, 'rb') as img:
            orgY = []
            unfiltY = []
            width = 12
            height = 12
            block = width * height

            initSize = '<' + str(block) + 'B'
            strSize = struct.calcsize(initSize)
            data = img.read(strSize)

            # 하나의 bin file 안에 unfiltered, original 순서로 data 담겨있음
            if idx % 2 == 0:
                unfiltY.append(struct.unpack(initSize, data))
            else:
                orgY.append(struct.unpack(initSize, data))

            orgY = np.array(orgY).reshape(12, 12, 1).transpose(1, 2, 0)
            unfiltY = np.array(unfiltY).reshape(12, 12, 1).transpose(1, 2, 0)

            return orgY, unfiltY


    # 다음 data 읽어옴
    def getNextData(self):
        if self.iter >= self.dataSize:
            self.iter = 0

        orgData = self.initTrainingSet['org'][self.dataOrder[self.iter]]
        unfiltData = self.initTrainingSet['unfilt'][self.dataOrder[self.iter]]
        resData = self.initTrainingSet['res'][self.dataOrder[self.iter]]

        self.iter += 1

        return orgData, unfiltData, resData



class Tester(object):
    batch_size = 1
    TestSetPath = "./Dataset/Test"

    def __init__(self):
        self.testSetList = self.getTestDatalist(Tester.TestSetPath)
        logger.info("Test Set Build Finished : %d" )
        self.L1_loss, self.L2_loss, self.CL2_Loss = resultNetwork(reuse=True, name='Common', training=False)
        logger.info("Test Graph Build Finished")


    def getTestDatalist(self, path, pattern='.bin'):
        sequences = []

        for root, dirNames, fileNames in os.walk(path):
            for fileName in fileNames:
                if fileName.endswith(pattern):
                    sequences.append(os.path.join(root, fileName))

        return sequences


    def setTestData(self):
        dataset = {'org' : [], 'unfilt' : [], 'res' : []}

        for fileList in self.testSetList:
            fileList = fileList[:200]
            fileList = np.array(fileList).reshape((-1, 1))

            for batchList in fileList:
                org = []
                unfilt = []
                res = []

                for idx, filePath in enumerate(batchList):
                    orgData, unfiltData = self.unpackTestSet(idx, filePath)

                    org.append(orgData)
                    unfilt.append(unfiltData)
                    res.append(orgData[idx] - unfiltData[idx])

                dataset['org'].append(np.array(org))
                dataset['unfilt'].append(np.array(unfilt))
                dataset['res'].append(np.array(res))

        return dataset, len(dataset['org'])


    def unpackTestSet(self, idx, imgPath):
        with open(imgPath, 'rb') as img:
            orgY = []
            unfiltY = []
            width = 12
            height = 12
            block = width * height

            initSize = '<' + str(block) + 'B'
            strSize = struct.calcsize(initSize)
            data = img.read(strSize)

            # 하나의 bin file 안에 unfiltered, original 순서로 data 담겨있음
            if idx % 2 == 0:
                unfiltY.append(struct.unpack(initSize, data))
            else:
                orgY.append(struct.unpack(initSize, data))

            orgY = np.array(orgY).reshape(12, 12, 1).transpose(1, 2, 0)
            unfiltY = np.array(unfiltY).reshape(12, 12, 1).transpose(1, 2, 0)

            return orgY, unfiltY


if __name__=='__main__':
    # test = Tester()

    train = Trainer()
    objectEpoch = 10
    modelNum = 0
    objectIter = objectEpoch * train.dataSize
    writeOp = tf.summary.merge_all()
    saver = tf.train.Saver()
    savePath = modelPath

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, savePath)
        orgData = tf.get_default_graph().get_tensor_by_name('Common_scope/orgData:0')
        unfiltData = tf.get_default_graph().get_tensor_by_name('Common_scope/unfiltData:0')
        resData = tf.get_default_graph().get_tensor_by_name('Common_scope/resData:0')

        for i in range(objectIter):
            org, unfilt, res = train.getNextData()
            _, _loss, _Closs = sess.run([train.opt, train.L2_loss, train.CL2_Loss],
                                        feed_dict={orgData:org, unfiltData:unfilt, resData:res})

            _loss = getPsnr(_loss)
            _Closs = getPsnr(_Closs)
            summary = sess.run(writeOp, {train.log:_loss})
            train.writer.add_summary(summary, i//10)
            summary = sess.run(writeOp, {train.log:_Closs})
            train.writer.add_summary(summary, i//10)

            if i % 50 == 0 or i == objectIter - 1:
                train.writer.flush()
                train.Cwriter.flush()
                logger.info("[CNN PSNR : %s]   [VVC PSNR : %s]" %(_loss, _Closs))
                modelPath = modelPath + 'model_' + str(modelNum) + '.ckpt'
                modelNum += 1
                savePath = saver.save(sess, modelPath)
                logger.info("Model saved in file %s" %savePath)







