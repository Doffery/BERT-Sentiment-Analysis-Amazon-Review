'''
Code from Mo.
'''

# Call the command line output from here and read the output file
import subprocess,os, json
import numpy as np
import tensorflow as tf
from random import randint
# Try running command from here

#p = subprocess.Popen('python ../bert-master/extract_features.py \
# --input_file="./newsSmallTesting.txt" --output_file="./newsSmallTesting12layers.jsonl" \
# --vocab_file="../uncased_L-12_H-768_A-12/vocab.txt" \
# --bert_config_file="../uncased_L-12_H-768_A-12/bert_config.json" \
# --init_checkpoint="../uncased_L-12_H-768_A-12/bert_model.ckpt" \
# --layers=-1,-2,-3,-4 --max_seq_length=128 --batch_size=8',stdout=subprocess.PIPE)
#p.wait()
# Should be a parameter
numLayers = 1
embMode = "MAX"
outputFile = "./BertVectors/SampleVec.txt"
trainFileName='./data/results_judged_task_v2.0_train_bert_word_encoding_1.jsonl'
testFileName='./data/results_judged_task_v2.0_test_bert_word_encoding_1.jsonl'
testResultFileName='./data/testResult.tsv'
prFileName='./data/PR.tsv'
printTokenization = 1

batchSize = 48
lstmUnits = 64
numClasses = 3
iterations = 1000
maxSeqLength=100
numDimensions=768  #BERT base 1 layer
trainingDataSize=4112
numLayers=1
testDataSize=1028
isTest=1

def handleNumLayers(numLayers,jsonObject):
    if numLayers == 1:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values']
    elif numLayers == 2:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values'] + \
                jsonObject['layers'][1]['values']
    elif numLayers == 3:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values'] + \
                jsonObject['layers'][1]['values'] + \
                jsonObject['layers'][2]['values']
    elif numLayers == 4:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values'] + \
                jsonObject['layers'][1]['values'] + \
                jsonObject['layers'][2]['values'] + \
                jsonObject['layers'][3]['values']
    else:
        print('Number of layers parameter not set to a valid value\n')
        return []
    return vecSent


def genSentenceEmbedding(data,embMode,numLayers):
    vecSent = []    
    numWords = len(data['features'])
    
    if 'linex_index' in data.keys():
        line_index = data['linex_index']
    elif 'line_index' in data.keys():
        line_index = data['line_index']
    else:
        print('Line index not found')
        return ""

    if embMode == "SEP":
        #Extract the embedding of "SEP" token -- "SEP" is the last token in each sentence
        vecSentL = data['features'][numWords-1]
        vecSent = handleNumLayers(numLayers,vecSentL)
    elif embMode == "CLS":
        vecSentL = data['features'][0]
        vecSent = handleNumLayers(numLayers,vecSentL)
    elif embMode == "AVG":
        for index in range(1,numWords-1): # exclude the CLS & the SEP token
            vecWordL = data['features'][index]
            vecWordL = handleNumLayers(numLayers,vecWordL)
            if index == 1:
                vecSent = vecWordL
            else:
                vecSent = np.add(vecSent,vecWordL)
        vecSent = vecSent/ (index - 2) #excluding the first and the last word
    elif embMode == "MAX":
        for index in range(1,numWords-1): # exclude the CLS & the SEP token
            vecWordL = data['features'][index]
            vecWordL = handleNumLayers(numLayers,vecWordL)
            if index == 1:
                vecSent = vecWordL
            else:
                vecSent = np.maximum(vecSent,vecWordL)
    else:
        print('The sentence embedding mode was not entered correctly')

    # Generate the tokenized version as well
    tokenized = ""
    for index in range(0,numWords): #exclude the CLS & the SEP token
            tokenObj = data['features'][index]
            tokenized = tokenized + " " + tokenObj['token']
    return vecSent, tokenized.strip(),line_index

if isTest == 0:
    #get train data cube
    dataCube = np.zeros([trainingDataSize, maxSeqLength, numDimensions],dtype=float)
    labelMatrix = np.zeros([trainingDataSize],dtype=float)

    numWords = 0

    with open(trainFileName, encoding='utf-8') as data_file:
        lineIdx=0
        for line in data_file.readlines() :
            data = json.loads(line)
            numWords = maxSeqLength if (len(data['features'])>maxSeqLength) else len(data['features'])
            
            for index in range(0,numWords):
                vecWordL = data['features'][index]
                dataCube[lineIdx][index]=handleNumLayers(numLayers,vecWordL)

            if 'linex_index' in data.keys():
                line_index = data['linex_index']
            elif 'line_index' in data.keys():
                line_index = data['line_index']
            else:
                print('Line index not found')
            
            labelMatrix[lineIdx]=int(line_index.split(';;')[1])
            lineIdx=lineIdx+1

if isTest == 1:
    #get test data cube
    testDataCube = np.zeros([trainingDataSize, maxSeqLength, numDimensions],dtype=float)
    testLabelMatrix = np.zeros([trainingDataSize],dtype=float)
    numWords = 0
    testText=[]

    with open(testFileName, encoding='utf-8') as data_file:
        lineIdx=0
        for line in data_file.readlines() :
            data = json.loads(line)
            numWords = maxSeqLength if (len(data['features'])>maxSeqLength) else len(data['features'])
            testSeq=""
            for index in range(0,numWords):
                vecWordL = data['features'][index]
                testDataCube[lineIdx][index]=handleNumLayers(numLayers,vecWordL)
                testSeq=testSeq+' '+vecWordL['token']

            if 'linex_index' in data.keys():
                line_index = data['linex_index']
            elif 'line_index' in data.keys():
                line_index = data['line_index']
            else:
                print('Line index not found')
            
            testLabelMatrix[lineIdx]=int(line_index.split(';;')[1])
            testText.append(testSeq)
            lineIdx=lineIdx+1



def getTrainBatch():
    labels = []
    
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])
    for i in range(batchSize):
        num = randint(1,trainingDataSize)
        label=labelMatrix[num-1:num]

        if label==2:
            labels.append([1,0,0])
        elif label==1:
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])
        
        arr[i] = dataCube[num-1:num]
    return arr, labels


def getTestBatch(index):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])

    if index*batchSize>testDataSize:
        return arr, labels

    endIdx=0
    if (index+1)*batchSize>testDataSize:
        endIdx=testDataSize
    else:
        endIdx=(index+1)*batchSize
    arrIdx=0
    for i in range(index*batchSize, endIdx):
        num = i+1
        label=testLabelMatrix[num-1:num]

        if label==2:
            labels.append([1,0,0])
        elif label==1:
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])
        
        arr[arrIdx] = testDataCube[num-1:num]
        arrIdx=arrIdx+1

    return arr, labels


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
data = tf.placeholder(tf.float32, [batchSize, maxSeqLength, numDimensions])

from tensorflow.contrib import rnn

#one direction LSTM
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

#bidirectional LSTM
#right_lstm_cell = rnn.BasicLSTMCell(num_units=lstmUnits)
#left_lstm_cell = rnn.BasicLSTMCell(num_units=lstmUnits)

#right_lstm_cell = rnn.DropoutWrapper(right_lstm_cell,output_keep_prob=0.75)
#left_lstm_cell = rnn.DropoutWrapper(left_lstm_cell,output_keep_prob=0.75)
#l2_loss = tf.constant(value=0.0,dtype=tf.float32)


#value,state = tf.nn.bidirectional_dynamic_rnn(left_lstm_cell,right_lstm_cell,data,dtype=tf.float32)
#combined_output = tf.concat(value, axis=2)
#weight = tf.Variable(tf.truncated_normal([lstmUnits*2, numClasses]))
#bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
#value = tf.transpose(combined_output, [1, 0, 2])
#last = tf.gather(value, int(value.get_shape()[0]) - 1)
#prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)



import datetime

sess = tf.InteractiveSession()

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)




saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if isTest == 0:
    for i in range(iterations):
        #Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch()
        sess.run(optimizer, {data: nextBatch, labels: nextBatchLabels})
        #Write summary to Tensorboard
        if (i % 10 == 0):
            summary = sess.run(merged, {data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)
        #Save the network every 10,000 training iterations
        if (i % 100 == 0 and i != 0):
           save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
           print("saved to %s" % save_path)
    writer.close()

elif isTest==1:
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))

    numTestBatches=testDataSize//batchSize
    results = []
    pred=tf.nn.softmax(prediction,1)
    #pred=tf.argmax(prediction,1)

    for i in range(0,numTestBatches):
        nextBatch, nextBatchLabels = getTestBatch(i)
        p = sess.run(pred,{data: nextBatch, labels: nextBatchLabels})
        results.append(p)

    pp=0
    nn=0
    nunu=0
    pn=0
    pnu=0
    np=0
    nnu=0
    nup=0
    nun=0
    with open(testResultFileName,"w", encoding='utf-8') as result_file:
        batchId=0
        for batchResult in results:
            for idx in range(0,batchSize):
                if batchSize*batchId+idx>=testDataSize:
                    break
                seq = testText[batchSize*batchId+idx]
                truelabel=testLabelMatrix[batchSize*batchId+idx]
                result = batchResult[idx]
                seq = seq + '\t' + " ".join(str(x) for x in result)
                maxIdx=1
                if result[0]>=result[1] and result[0]>=result[2]:
                    maxIdx=0
                if result[1]>=result[0] and result[1]>=result[2]:
                    maxIdx=1
                if result[2]>=result[1] and result[2]>=result[0]:
                    maxIdx=2
                label=1
                if maxIdx==0:
                    label=2
                    if truelabel==2:
                        pp=pp+1
                    elif truelabel==1:
                        pnu=pnu+1
                    elif truelabel==0:
                        pn=pn+1
                elif maxIdx==1:
                    label=1
                    if truelabel==2:
                        nup=nup+1
                    elif truelabel==1:
                        nunu=nunu+1
                    elif truelabel==0:
                        nun=nun+1
                elif maxIdx==2:
                    label=0
                    if truelabel==2:
                        np=np+1
                    elif truelabel==1:
                        nnu=nnu+1
                    elif truelabel==0:
                        nn=nn+1
                seq=seq+'\t'+str(label)+'\t'+str(truelabel)
                result_file.writelines(seq)
                result_file.writelines('\n')
            
            batchId=batchId+1

        result_file.close()



    with open(prFileName,"w", encoding='utf-8') as pr_file:
        pr_file.writelines("positive positive\t"+str(pp))
        pr_file.writelines('\n')
        pr_file.writelines("positive neutral\t"+str(pnu))
        pr_file.writelines('\n')
        pr_file.writelines("positive negative\t"+str(pn))
        pr_file.writelines('\n')
        pr_file.writelines("neutral positive\t"+str(nup))
        pr_file.writelines('\n')
        pr_file.writelines("neutral neutral\t"+str(nunu))
        pr_file.writelines('\n')
        pr_file.writelines("neutral negative\t"+str(nun))
        pr_file.writelines('\n')
        pr_file.writelines("negative positive\t"+str(np))
        pr_file.writelines('\n')
        pr_file.writelines("negative neutral\t"+str(nnu))
        pr_file.writelines('\n')
        pr_file.writelines("negative negative\t"+str(nn))
        pr_file.writelines('\n')

    pr_file.close()
