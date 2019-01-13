# Call the command line output from here and read the output file
import subprocess,os, json
import numpy as np
# Try running command from here
# p = subprocess.Popen('python extract_features_n.py --input_file="./input.txt" --output_file="./output_2.jsonl" --vocab_file="../uncased_L-12_H-768_A-12/vocab.txt" --bert_config_file="../uncased_L-12_H-768_A-12/bert_config.json" --init_checkpoint="../uncased_L-12_H-768_A-12/bert_model.ckpt" --layers=-1,-2,-3,-4 --max_seq_length=128 --batch_size=8',stdout=subprocess.PIPE, shell=True)
# p.wait()
# Should be a parameter
numLayers = 2
embMode = "MAX"
numSent = 2
outputFile = "./BertVectors/SampleVec.txt"
printTokenization = 1

def handleNumLayers(numLayers,jsonObject):
    if numLayers == 1:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values']
    elif numLayers == 2:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values'] + jsonObject['layers'][1]['values']
    elif numLayers == 3:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values'] + jsonObject['layers'][1]['values'] + jsonObject['layers'][2]['values']
    elif numLayers == 4:
        # Take only the last 2 layers
        vecSent = jsonObject['layers'][0]['values'] + jsonObject['layers'][1]['values'] + jsonObject['layers'][2]['values'] + jsonObject['layers'][3]['values']
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

if __name__ == '__name__':

    #Open the file and process to aggregate word vectors to get the sentence vectors
    print('Load the json object and write the wrapper')
    with open(outputFile, encoding='utf-8',mode='w') as write_file:
        with open('output_2.jsonl', encoding='utf-8') as data_file:
            for line in data_file.readlines() :
                data = json.loads(line)
                vecSent,tokenizedSent,index = genSentenceEmbedding(data,embMode,numLayers)
                print(vecSent)
                print(tokenizedSent)
                print(index)
                id = index.split(';;')[0]
                label = index.split(';;')[1]
                if printTokenization == 1:
                    write_file.write(id + "\t" + label + "\t" + tokenizedSent + "\t" ) #" ".join(str(elem) for elem in vecSent))
                else:
                    write_file.write(id + "\t" + label + "\t" ) #" ".join(str(elem) for elem in vecSent))
                write_file.write(np.array2string(vecSent, precision=4, separator=' ',suppress_small=True))
                write_file.write("\n")

    #Extract the embeddings for each word
    print('Options entered: 1) Sentence Embedding Mode: ' + embMode + ' 2) Layers to be considered are: ' + str(numLayers))
    print('The length of the vector being output is: ' + str(len(vecSent)))
