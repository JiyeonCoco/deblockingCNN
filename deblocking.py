import os
import struct
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import logging
import time


TEMP_PATH = './temp_jy'
MODEL_PATH = './model_jy/'
TENSORBOARD_LOG_PATH = './tensorboard_jy/'
TRAINING_SET_PATH = './Dataset/Training'
TEST_SET_PATH = './Dataset/Test'

FILTER_NUM = 12
LEARNING_RATE = 1e-4
EPSILON = 1e-4  # AdamOptimizer epsilon

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

TARGET_DATA_NUM = 300000
QP_LIST = ['22', '32']

# msg.log 파일에 log 기록
class LoggingHelper(object):
    INSTANCE = None

    def __init__(self):
        if LoggingHelper.INSTANCE is not None:
            raise ValueError("An instantiation already exists!")

        os.makedirs(TEMP_PATH, exist_ok=True)
        self.logger = logging.getLogger()

        logging.basicConfig(filename='LOGGER', level=logging.DEBUG)

        fileHandler = logging.FileHandler(TEMP_PATH + '/msg.log')
        streamHandler = logging.StreamHandler()

        formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

    @classmethod
    def getInstace(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = LoggingHelper()
        return cls.INSTANCE

    @staticmethod
    def diff_time_logger(message, startTime):
        LoggingHelper.getInstace().logger.info("[{}] :: running time {}".format(message, time.time() - startTime))

logger = LoggingHelper.getInstace().logger


def getPsnr(loss):
    if loss <= 0:
        return 100
    return math.log10(1/loss) * 10


def getPSNRfromNumpy(A, B):
    return math.log10( (255.0 * 255.0) / np.square(np.subtract(A,B)).mean() ) * 10


def concatenation(layers):
    return tf.concat(layers, axis=-1)


def conv2d(inputX, filterNum, kernelSize, stride=1, rate=1, name='conv', padding='SAME', activation=tf.nn.relu, training=True):
    assert padding in ['SYMMETRIC', 'VALID', 'SAME', 'REFLECT']

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int(rate * (kernelSize - 1) / 2)
        inputX = tf.pad(inputX, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'

        inputX = tf.layers.conv2d(inputX, filterNum, kernelSize, stride, dilation_rate=rate, activation=activation, padding=padding, name=name)

    return inputX



def batchActivConv(inputX, filterNum, kernelSize, training, activation=None, rate=1, name="layer"):
    with tf.variable_scope(name):
        inputX = tf.contrib.layers.batch_norm(inputX, scale=True, training=training, updates_collections=None)
        inputX = tf.nn.relu(inputX)
        inputX = conv2d(inputX, filterNum, kernelSize, stride=rate, activation=activation)

        return inputX


def relu(inputX):
    return tf.nn.relu(inputX)


def batchNormalization(inputX, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=inputX, training=training, reuse=None),
                       lambda: batch_norm(inputs=inputX, training=training, reuse=True))


def bottleneckLayer(inputX, scope, training, filterNum):
    with tf.name_scope(scope):
        inputX = tf.contrib.layers.batch_norm(inputX, scale=True, training=training, updates_collections=None)
        inputX = relu(inputX)
        inputX = conv2d(inputX, 4*filterNum, 1, name=scope+'_conv1')

        inputX = tf.contrib.layers.batch_norm(inputX, scale=True, training=training, updates_collections=None)
        inputX = relu(inputX)
        inputX = conv2d(inputX, filterNum, 3, name=scope+'_conv2')

        return inputX


def denseBlock(inputX, layerNum, training, layerName):
    with tf.name_scope(layerName):
        layers_concat = list()
        layers_concat.append(inputX)

        x = bottleneckLayer(inputX, scope=layerName+'_bottleN_'+str(0), training=training, filterNum=FILTER_NUM)

        for i in range(layerNum - 1):
            x = concatenation(layers_concat)
            x = bottleneckLayer(x, scope=layerName + '_bottleN'+str(i+1), training=training, filterNum=FILTER_NUM)
            layers_concat.append(x)

        x = concatenation(layers_concat)

        return x


def mainNetwork(inputX, training, layerNum=5, reuse=False, name='Separable'):
    with tf.variable_scope(name + "_scope", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variable()

        # [convolution 1] filter: 64  /  kernel size: 3x3  /  padding O
        # [convoltuion 2] filter: 32  /  kernel size: 3x3  /  padding O
        conv1 = conv2d(inputX, 64, 3, padding='SAME', name=name+'conv1')
        conv2 = conv2d(conv1, 32, 3, padding='SAME', name=name+'conv2')
        concatLayers = list()
        concatLayers.append(conv2)

        # convolution 2 결과를 가지고 bottleneck layer 쌓아서 convolution (filter: 12*4 -> 12)
        for i in range(layerNum - 1):
            convX = concatenation(concatLayers)
            convX = bottleneckLayer(convX, scope=name+'_bottleN'+str(i + 1), training=training, filterNum=FILTER_NUM)
            concatLayers.append(convX)

        # [convolution 3] filter: 32  /  kernel size: 3x3  /  padding O
        convX = concatenation(concatLayers)
        concatLayers = list()
        convX = tf.layers.conv2d(convX, 32, [3, 3], padding='SAME', name=name+'conv3')
        concatLayers.append(convX)

        # convolution 3 결과를 가지고 bottleneck layer 쌓아서 convolution (filter: 12*4 -> 12)
        for i in range(layerNum - 1):
            convX = concatenation(concatLayers)
            convX = bottleneckLayer(convX, scope=name+'_bottleN2'+str(i + 1), training=training, filterNum=FILTER_NUM)
            concatLayers.append(convX)

        # [convolution 4] filter: 1  /  kernel size: 3x3  /  padding O
        convX = concatenation(concatLayers)
        convX = tf.layers.conv2d(convX, 1, [3, 3], padding='SAME', name=name + 'conv4')

        return convX


def resultNetwork(reuse=False, name='Common', training=True):
    with tf.variable_scope(name+"_scope", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()

        orgData = tf.placeholder(tf.float32, shape = [None, None, None, 1], name="orgData")/255.
        unfiltData = tf.placeholder(tf.float32, shape = [None, None, None, 3], name="unfiltData")/255.
        filtData = tf.placeholder(tf.float32, shape = [None, None, None, 1], name='filtData')/255.

        # unfiltered data를 가지고 training 및 test (mainNetwork)
        result = mainNetwork(unfiltData, training, name=name+"Net", layerNum=8)
        unfiltData = tf.slice(unfiltData, [0,0,0,0], [tf.shape(unfiltData)[0], tf.shape(unfiltData)[1], tf.shape(unfiltData)[2], 1])
        trainedData = tf.math.add(unfiltData, result)

        # original data와 trained data의 loss 계산 (L1_loss, L2_loss)
        # 비교대상은 original data와 기존의 HEVC 방식을 거친 뒤의 data의 loss (base_loss)
        L1_loss = tf.reduce_mean(tf.losses.absolute_difference(labels=orgData, predictions=trainedData))
        L2_loss = tf.losses.mean_squared_error(labels=orgData, predictions=trainedData)
        base_loss = tf.losses.mean_squared_error(labels=orgData, predictions=filtData)

        return L1_loss, L2_loss, base_loss


class Trainer(object):
    def __init__(self, name="Train", path=TRAINING_SET_PATH, reuse=False, training=True, batchSize=6, targetNum=TARGET_DATA_NUM):
        self.iter = 0
        self.dataNum = 0
        self.dataset = { 'org' : [], 'unfiltered' : [], 'filtered' : [] }
        self.batchSize = batchSize

        self.trainingSetList = self.getTrainingDataList(path)
        self.setTrainingData(targetNum)

        self.dataSize = len(self.dataset['org'])
        self.dataOrder = np.arange(self.dataSize)
        np.random.shuffle(self.dataOrder)

        logger.info("TrainingSet Build Finished")

        self.L1_loss, self.L2_loss, self.base_loss = resultNetwork(reuse=reuse, name='Common', training=training)

        if reuse == False:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=EPSILON)
            self.opt = self.optimizer.minimize(self.L1_loss)

        self.writer = tf.summary.FileWriter(TENSORBOARD_LOG_PATH + name + '_CNN_Prediction')
        self.baseWriter = tf.summary.FileWriter(TENSORBOARD_LOG_PATH + name + '_HEVC_Prediction')
        self.log = tf.Variable(0.0)
        self.summary = tf.summary.scalar(name+"_Sequences", self.log)

        logger.info("Graph Build Finished")


    # 디렉토리 내에 있는 bin file 차례대로 읽어서 구성 (sequences)
    def getTrainingDataList(self, path, pattern='.bin'):
        sequences = []

        for root, dirNames, fileNames in os.walk(path):
            for fileName in fileNames:
                if fileName.endswith(pattern):
                    sequences.append(os.path.join(root, fileName))

        return sequences


    # training dataset을 batch size에 맞춰 구성
    # trainingSetList : bin file 개수  /  targetNum : training에 사용할 data 개수
    def setTrainingData(self, targetNum):
        getFilenumEach = targetNum // len(self.trainingSetList)

        for fileList in self.trainingSetList:
            self.unpackTrainingData(fileList, getFilenumEach)
        return


    # bin file로부터 data 읽어서 각각 구성 (original, unfiltered)
    def unpackTrainingData(self, imgPath, targetNum):
        with open(imgPath, 'rb') as img:
            qp = int(imgPath.split('_')[-1].split('.')[0])
            size = int(imgPath.split('_')[-2].split('x')[0])

            width = size + 4
            height = size + 4
            block = width * height
            initSize = '<' + str(block) + 'B'

            # fileSize  : bin file 내의 data 개수
            # block * 3 : original, unfiltered, filtered data를 한 쌍으로 만들어서 data random shuffle
            # 즉, fileSize는 block의 총 개수
            fileSize = os.path.getsize(imgPath)
            fileSize //= (block * 3)
            randomArr = np.arange(0, fileSize)
            np.random.shuffle(randomArr)
            randomArr = randomArr[:targetNum]

            orgY = []
            unfiltY = []
            filtY = []

            for i in range(fileSize):
                if i in randomArr:
                    # batch size만큼 구성되면 값 array에 삽입
                    if len(orgY) == self.batchSize:
                        orgY = np.array(orgY)
                        unfiltY = np.array(unfiltY)
                        filtY = np.array(filtY)
                        # block size와 동일한 크기의 QP mask 생성 (값은 0으로 채움)
                        qpMask = np.zeros((self.batchSize, height, width, len(QP_LIST)))

                        if qp == QP_LIST[0]:
                            qpMask[:, :, :, 0] += 255
                        elif qp == QP_LIST[1]:
                            qpMask[:, :, :, 1] += 255

                        # training을 위해 unfiltered data와 QP mask를 concatenate (unfilted data 뒤에 QP mask)
                        unfiltY = np.concatenate((unfiltY, qpMask), axis=-1)

                        self.dataset['org'].append(orgY)
                        self.dataset['unfiltered'].append(unfiltY)
                        self.dataset['filtered'].append(filtY)

                        orgY = []
                        unfiltY = []
                        filtY = []

                    orgY.append(np.array(struct.unpack(initSize, img.read(block))).reshape(height, width, 1))
                    unfiltY.append(np.array(struct.unpack(initSize, img.read(block))).reshape(height, width, 1))
                    filtY.append(np.array(struct.unpack(initSize, img.read(block))).reshape(height, width, 1))

            logger.info("PSNR %s" % (getPSNRfromNumpy( np.array(self.dataset['org'][self.dataNum:]), np.array(self.dataset['unfilt'][self.dataNum:])[:,:,:,:,0][:,:,:,:,np.newaxis] )))
            logger.info("[%s : cnt(%d/%d)]" % (os.path.basename(imgPath), targetNum, fileSize))

            # batch size만큼 끊어서 data number 갱신 (PSNR 구하기 위해)
            self.dataNum += targetNum // self.batchSize


    # 다음 data 읽어옴
    def getNextData(self):
        if self.iter >= self.dataSize:
            self.iter = 0
            logger.info('Finished Epoch')

        orgData = self.dataset['org'][self.dataOrder[self.iter]]
        unfiltData = self.dataset['unfiltered'][self.dataOrder[self.iter]]
        filtData = self.dataset['filtered'][self.dataOrder[self.iter]]

        self.iter += 1

        return orgData, unfiltData, filtData


# tensor 실행
if __name__=='__main__':
    train = Trainer()
    test = Trainer(path=TEST_SET_PATH, name="Test", training=False, reuse=True, batchSize=1, targetNum=30000)
    objectEpoch = 10
    modelNum = 0
    objectIter = objectEpoch * train.dataSize
    writeOp = tf.summary.merge_all()
    saver = tf.train.Saver()
    modelPath = MODEL_PATH
    logger.info('Oeject iteration : %s' %objectIter)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # for training
        orgData = tf.get_default_graph().get_tensor_by_name('Common_scope/orgData:0')
        unfiltData = tf.get_default_graph().get_tensor_by_name('Common_scope/unfiltData:0')
        filtData = tf.get_default_graph().get_tensor_by_name('Common_scope/filtData:0')

        # for test
        t_orgData = tf.get_default_graph().get_tensor_by_name('Common_scope_1/orgData:0')
        t_unfiltData = tf.get_default_graph().get_tensor_by_name('Common_scope_1/unfiltData:0')
        t_filtData = tf.get_default_graph().get_tensor_by_name('Common_scope_1/filtData:0')

        for i in range(1, objectIter+1):
            org, unfilt, filt = train.getNextData()
            _, _loss, _baseLoss = sess.run([train.opt, train.L2_loss, train.base_loss],
                                        feed_dict={ orgData: org, unfiltData: unfilt, filtData: filt })

            _loss = getPsnr(_loss)
            _baseLoss = getPsnr(_baseLoss)
            summary = sess.run(writeOp, {train.log: _loss})
            train.writer.add_summary(summary, i)
            summary = sess.run(writeOp, {train.log: _baseLoss})
            train.baseWriter.add_summary(summary, i)

            # 10000번째마다, training log 출력 및 training model 저장
            if i % 10000 == 0 or i == objectIter:
                # 300000번째 training 때 learning rate 줄임 (학습률 향상을 위해)
                if i % 300000 == 0:
                    LEARNING_RATE /= 10

                train.writer.flush()
                train.baseWriter.flush()
                logger.info("(%s / %s)[CNN PSNR : %s]   [VVC PSNR : %s]    [Model Saved : %s]" %(i, objectIter, _loss, _baseLoss, modelPath))
                # training model 저장
                modelPath = MODEL_PATH + 'model_' + str(modelNum) + '.ckpt'
                modelNum += 1
                savePath = saver.save(sess, modelPath)
                logger.info("Model saved in file %s" %savePath)

                # 30000번째마다, test log 출력
                if i % 30000 == 0 or i == objectIter:
                    mean_loss = []
                    mean_baseLoss = []

                    for i in range(test.dataSize):
                        org, unfilt, filtered = test.getNextData()

                        _loss, _baseLoss = sess.run([test.L2_loss, test.base_loss],
                                                    feed_dict={ t_orgData: org, t_unfiltData: unfilt, t_filtData: filt })

                        mean_loss.append(_loss)
                        mean_baseLoss.append(_baseLoss)

                    test_psnr = getPsnr(np.array(mean_loss).mean())
                    hevc_psnr = getPsnr(np.array(mean_baseLoss).mean())

                    summary = sess.run(writeOp, {test.log: test_psnr})
                    test.writer.add_summary(summary, i)
                    summary = sess.run(writeOp, {test.log: hevc_psnr})
                    test.baseWriter.add_summary(summary, i)

                    test.writer.flush()
                    test.baseWriter.flush()
                    logger.info("[TestPSNR : %s] [HEVC : %s]" %(test_psnr, hevc_psnr))