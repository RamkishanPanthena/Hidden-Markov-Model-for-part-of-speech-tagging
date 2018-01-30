import pandas as pd
import sys

def viterbi_algo(sent):
    global vocab
    global a
    global b
    
    sent = sent.lower().split()
    viterbi = []
    backpointer = []
    
    if sent[0] in vocab:
        viterbi.append((a.ix['*']*b[sent[0]]).max(0))
        backpointer.append((a.ix['*']*b[sent[0]]).argmax(0))
    else:
        viterbi.append((a.ix['*']*b['<unk>']).max(0))
        backpointer.append((a.ix['*']*b['<unk>']).argmax(0))
    
    for i in range(1,len(sent)):
        if sent[i] in vocab:
            viterbi.append((a.ix[backpointer[i-1]]*b[sent[i]]).max(0))
            backpointer.append((a.ix[backpointer[i-1]]*b[sent[i]]).argmax(0))
        else:
            viterbi.append((a.ix[backpointer[i-1]]*b['<unk>']).max(0))
            backpointer.append((a.ix[backpointer[i-1]]*b['<unk>']).argmax(0))
        
    return(backpointer)


#base = 'C:/Users/Krishna/Desktop/Data Science/Northeastern University/NEU/Study Material/CS 6120 - Natural Language Processing/Assignments/HW1/a1_datasets/q4_UD_English/'

inputdata = sys.argv
train = inputdata[1]
test = inputdata[2]

#train_file = base+'train.counts'
train_file = train

word_tag_counts = dict()
tag_counts = dict()
tag1_tag2_counts = dict()
emission_probability = dict()
transition_probability = dict()
emission_word_to_tag_map = dict()
transition_tag_prevtag_map = dict()
vocab = set()

print("Loading the training set..")
# Load the train data
with open(train_file) as f:
    line = f.read().splitlines()

print("Getting counts of words, tags..")
# Get count of unigram tags
for i in range(len(line)):
    k = line[i].split()
    if k[1] == 'WORDTAG':
        if int(k[0]) < 5:
            if (k[2], '<unk>') in word_tag_counts:
                t = word_tag_counts[(k[2], '<unk>')]
                word_tag_counts[(k[2], '<unk>')] = t+int(k[0])
            else:
                word_tag_counts[(k[2], '<unk>')] = int(k[0])
        else:
            word_tag_counts[(k[2],k[3].lower())] = int(k[0])
            vocab.add(k[3].lower())
    if k[1] == '1-GRAM':
        tag_counts[k[2]] = int(k[0])
    if k[1] == '2-GRAM':
        tag1_tag2_counts[(k[2],k[3])] = int(k[0])

# Calculate vocab size
vocab.add('<unk>')
V = len(vocab)

# Get count of start words
start_count = 0
stop_count = 0
for i in range(len(line)):
    k = line[i].split()
    if k[1] == '2-GRAM' and k[2] == '*':
        start_count+=int(k[0])
    if k[1] == '2-GRAM' and k[3] == 'STOP':
        stop_count+=int(k[0])
        
#tag_counts['*'] = start_count
#tag_counts['STOP'] = stop_count

# Create a dataframe with tags as rows
df1 = pd.DataFrame(index=tag_counts)

# Get column list
emission_col_list = []
for i in vocab:
    emission_col_list.append(i)

print('\n')    
print("Building emission matrix..")
# Build emission matrix
for i in emission_col_list:
    col = []
    for j in tag_counts:
        #print(j)    
        if (j,i) in word_tag_counts:
            col.append(word_tag_counts[(j, i)]/tag_counts[j])
        else:
            col.append(0)
    df1[i] = col

# Create a dataframe with unique bigram tags as rows
transition_row_list = ['*']
for i in tag_counts:
    transition_row_list.append(i)

df2 = pd.DataFrame(index=transition_row_list)

all_tag_counts = tag_counts.copy()
all_tag_counts['*'] = start_count

print('\n')
print("Build transition matrix..")
# Build transition matrix
for i in tag_counts:
    col = []
    for j in transition_row_list:
        if (j,i) in tag1_tag2_counts:
            col.append(tag1_tag2_counts[(j, i)]/all_tag_counts[j])
        else:
            col.append(0)
    df2[i] = col
    
a = df2 # Transition matrix
b = df1 # Observation matrix
###########################################################
print("Loading test data..")

#test_file = base+'test.words'
test_file = test

with open(test_file) as f:
    line = f.read().splitlines()


sentences = []
sent=''

for i in range(len(line)):
    #k = line[i].split()
    if line[i] == '':
        sentences.append(sent)
        sent=''
    else:
        sent+=line[i]+' '

print('\n')
print("Storing the sentences..")
sent_split = []
for i in range(len(sentences)):
    sent_split.append(sentences[i].split())


print('\n')
print("Calculating tags for sentences using Viterbi algorithm..")
calculated_tags_split = []
for i in range(len(sentences)):
    calculated_tags_split.append(viterbi_algo(sentences[i]))

#####################
print('\n')
print("Writing to a file..", inputdata[3])

outputfile = open(inputdata[3], 'a')

correct=0
total=0
for i in range(len(calculated_tags_split)):
    for j in range(len(calculated_tags_split[i])):
        tmp = sent_split[i][j]+' '+calculated_tags_split[i][j]
        outputfile.writelines(tmp+'\n')
    outputfile.writelines('\n')
    