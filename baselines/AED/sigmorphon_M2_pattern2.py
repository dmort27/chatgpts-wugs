
#!/usr/bin/env python
# coding: utf-8



from __future__ import print_function
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, RepeatVector, Activation, Flatten, Masking, Bidirectional
import numpy as np
import pandas as pd
import pickle


batch_size = 32  
epochs = 40  
#epochs = 20  
size_LSTM = 200 
#size_LSTM = 100 


##### load merging file to set the vectors length #####

data_path = "../data/2023InflectionST/patterns/eng.merged.phono.patterns"

poss_m = []
patterns_2_m_output = []
patterns_2_m_input = []
patterns_2_m = []
lemma_phonos_m = []
form_orthos_m = []
poss_type_m = set()
forms_characters_m = set()
pattern_2_m_output_characters_m = set()
pattern_2_m_input_characters_m = set()
lemma_phonos_characters_m = set()
form_orthos_characters_m = set()

lines = open(data_path, encoding='utf8').read().split('\n')
for line in lines[: min(len(lines), len(lines) - 1)]:
    line = line.split('\t')
    lemma_phono_m = "#" + line[0] + "$"
    pos_m = line[2]
    form_ortho_m = "#" + line[4] + "$"
    # pattern_2_m = line[6].split('/')
    pattern_2_m = line[8].split('/')
    pattern_2_m_input = "#" + pattern_2_m[0] + "$"
    pattern_2_m_output = "#" + pattern_2_m[1] + "$" 
    poss_m.append(pos_m)
    lemma_phonos_m.append(lemma_phono_m)
    form_orthos_m.append(form_ortho_m)
    patterns_2_m_input.append(pattern_2_m_input)
    patterns_2_m_output.append(pattern_2_m_output)
    for char in lemma_phono_m:
        if char not in lemma_phonos_characters_m:
            lemma_phonos_characters_m.add(char) 
    for tipo in poss_m:
        if tipo not in poss_type_m:
            poss_type_m.add(tipo)    
    for char in form_ortho_m:
        if char not in form_orthos_characters_m:
            form_orthos_characters_m.add(char)        
    for char in pattern_2_m_input:
        if char not in pattern_2_m_input_characters_m:
            pattern_2_m_input_characters_m.add(char)             
    for char in pattern_2_m_output:
        if char not in pattern_2_m_output_characters_m:
            pattern_2_m_output_characters_m.add(char)  

print("MERGING FILE LOADED")


##### load merging file to set the vectors length #####

data_path = "../data/2023InflectionST/patterns/eng.trn.phono.patterns"

poss = []
form_orthos = []
patterns_2_output = []
patterns_2_input = []
patterns_2 = []
lemma_phonos = []
poss_type = set()
pattern_2_output_characters = set()
pattern_2_input_characters = set()
lemma_phonos_characters = set()
form_orthos_characters = set()


lines = open(data_path, encoding='utf8').read().split('\n')
for line in lines[: min(len(lines), len(lines) - 1)]:
    line = line.split('\t')
    lemma_phono = "#" + line[0] + "$"
    pos = line[2]
    form_ortho = "#" + line[4] + "$"
    # pattern_2 = line[6].split('/')
    pattern_2 = line[8].split('/')
    pattern_2_input = "#" + pattern_2[0] + "$"
    pattern_2_output = "#" + pattern_2[1] + "$"
    lemma_phonos.append(lemma_phono)
    poss.append(pos)
    form_orthos.append(form_ortho)
    patterns_2_input.append(pattern_2_input)
    patterns_2_output.append(pattern_2_output)
    for tipo in poss:
        if tipo not in poss_type:
            poss_type.add(tipo)
    for char in form_ortho:
        if char not in form_orthos_characters:
            form_orthos_characters.add(char) 
    for char in lemma_phono:
        if char not in lemma_phonos_characters:
            lemma_phonos_characters.add(char)             
    for char in pattern_2_input:
        if char not in pattern_2_input_characters:
            pattern_2_input_characters.add(char)             
    for char in pattern_2_output:
        if char not in pattern_2_output_characters:
            pattern_2_output_characters.add(char)      

print("TRAIN FILE LOADED")


pattern_2_input_characters = sorted(list(pattern_2_m_input_characters_m))
pattern_2_output_characters = sorted(list(pattern_2_m_output_characters_m))
form_orthos_characters = sorted(list(form_orthos_characters_m))
lemma_phonos_characters = sorted(list(lemma_phonos_characters_m))
poss_type = sorted(list(poss_type_m))

print(" CHARACTERS =  MERGE CHARACTERS")


poss_type_index = {char: i for i, char in enumerate(poss_type)}
pattern_2_input_characters_index = {char: i for i, char in enumerate(pattern_2_input_characters)}
pattern_2_output_characters_index = {char: i for i, char in enumerate(pattern_2_output_characters)}
form_orthos_characters_index = {char: i for i, char in enumerate(form_orthos_characters)}
lemma_phonos_characters_index = {char: i for i, char in enumerate(lemma_phonos_characters)}
num_form_orthos_tokens = len(form_orthos_characters_index) 
num_lemma_phonos_tokens = len(lemma_phonos_characters_index) 
num_poss_type_tokens = len(poss_type_index)
num_pattern_2_input_characters_tokens = len(pattern_2_input_characters_index)
num_pattern_2_output_characters_tokens = len(pattern_2_output_characters_index)
max_poss_type_length = len(poss_type)
max_lemma_phonos_length = (max([len(txt) for txt in lemma_phonos]))
max_form_orthos_length = (max([len(txt) for txt in form_orthos]))
max_pattern_2_input_length = (max([len(txt) for txt in patterns_2_input]))
max_pattern_2_output_length = (max([len(txt) for txt in patterns_2_output]))


# encoding data in a 3d array

poss_data = np.zeros((len(poss), len(poss_type)),dtype='float32')
lemma_phonos_data = np.zeros((len(lemma_phonos), max_lemma_phonos_length, num_lemma_phonos_tokens), dtype='float32')
patterns_2_input_data = np.zeros((len(patterns_2_input), max_pattern_2_input_length, num_pattern_2_input_characters_tokens),dtype='float32')
patterns_2_output_data = np.zeros((len(patterns_2_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
decoder_input_data_patterns_2 = np.zeros((len(patterns_2_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
decoder_output_data_patterns_2 = np.zeros((len(patterns_2_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')

print("INPUT DATA CREATED")


print(poss_data.shape)
print(patterns_2_input_data.shape)
print(patterns_2_output_data.shape)
print(decoder_input_data_patterns_2.shape)
print(decoder_output_data_patterns_2.shape)


# one-hot encoding 

for i, (pattern_2_input, pattern_2_output, lemma_phono) in enumerate(zip(patterns_2_input, patterns_2_output, lemma_phonos)):
    for t, char in enumerate(pattern_2_input):
        patterns_2_input_data[i, t, pattern_2_input_characters_index[char]] = 1.  
    for t, char in enumerate(lemma_phono):
        lemma_phonos_data[i, t, lemma_phonos_characters_index[char]] = 1.          
    for t, char in enumerate(pattern_2_output):
        patterns_2_output_data[i, t, pattern_2_output_characters_index[char]] = 1.      
    for t, char in enumerate(pattern_2_output):
        decoder_input_data_patterns_2[i, t, pattern_2_output_characters_index[char]] = 1.
        if t > 0:
            decoder_output_data_patterns_2[i, t - 1, pattern_2_output_characters_index[char]] = 1.

for i,char in enumerate(poss):
        poss_data[i, poss_type_index[char]] = 1.


print("INPUT DATA VECTORIZATION CREATED")


#######################################
# NETWORK CONFIGURATION BIDIRECTIONAL #
#######################################
from tensorflow.keras.layers import TimeDistributed
from numpy import array

# input POS

input_POS = Input(shape=(num_poss_type_tokens,), name="input_POS")
input_POS_Repeat = RepeatVector(max_lemma_phonos_length)(input_POS)

# input lemma

encoder_lemma = Input(shape=(max_lemma_phonos_length, num_lemma_phonos_tokens), name="input_pattern2")

# merging POS + lemma

merging = Concatenate()([input_POS_Repeat, encoder_lemma])
masked_encoder_lemma = Masking(mask_value=0.)(merging)

# encoder lemma

encoder_lemma_BI_LSTM = Bidirectional(LSTM(size_LSTM, return_state=True), merge_mode="concat", name="encoder_lemma_BI_LSTM")
encoder_outputs_lemma, forward_h, forward_c, backward_h, backward_c = encoder_lemma_BI_LSTM(masked_encoder_lemma)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])


encoder_states_ALL = [state_h, state_h]

# decoder fap2

decoder_pattern2= Input(shape=(None, num_pattern_2_output_characters_tokens))
masked_decoder_pattern2 = Masking(mask_value=0.)(decoder_pattern2)
decoder_LSTM = LSTM(size_LSTM*2, return_sequences=True, return_state=True)
decoder_LSTM_outputs, _, _  = decoder_LSTM(masked_decoder_pattern2, initial_state = encoder_states_ALL)
decoder_outputs = Dropout(0.2)(decoder_LSTM_outputs)

# dense layer

decoder_dense = Dense(num_pattern_2_output_characters_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([input_POS, encoder_lemma, decoder_pattern2], decoder_outputs)

print("MODEL CREATED")

# run training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit([poss_data, lemma_phonos_data, decoder_input_data_patterns_2], decoder_output_data_patterns_2, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('2021Task0-main/part2/MODELS/model_sigmorpho_MODEL2_pattern2.h5')


###########################################
# INFERENCE ENCODER/DECODER BIDIRECTIONAL #
###########################################
# encoder

encoder_model = Model(inputs = [input_POS, encoder_lemma], outputs = encoder_states_ALL)
print("INFERENCE ENCODER LOADED")


# decoder

decoder_state_H = Input(shape=(size_LSTM*2,))
decoder_state_C = Input(shape=(size_LSTM*2,))
decoder_states_inputs = [decoder_state_H, decoder_state_C]
decoder_outputs, state_H, state_C = decoder_LSTM(decoder_pattern2, initial_state=decoder_states_inputs)
decoder_states = [state_H, state_C]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_pattern2] + decoder_states_inputs, [decoder_outputs] + decoder_states)
print("INFERENCE DECODER LOADED")


# sequence predicted and proba


reverse_output_pattern2_char_index = {i: char 
                            for char, i in pattern_2_output_characters_index.items()}

def sequence_out(pos, lemma_input_seq):
    states_value = encoder_model.predict([pos, lemma_input_seq])
    target_seq = np.zeros((1, 1, num_pattern_2_output_characters_tokens))
    target_seq[0, 0, pattern_2_output_characters_index['#']] = 1.

    stop_condition = False
    decoded_sentence_classic = ''
    decoded_sentence_argmax = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        proba = str(output_tokens[:, :, sampled_token_index])
        proba = proba[2:6]
        sampled_char = reverse_output_pattern2_char_index[sampled_token_index]
        decoded_sentence_classic += sampled_char
        decoded_sentence_argmax += ("[" + sampled_char + " " + proba + "]")
        if (sampled_char == '$' or
           len(decoded_sentence_classic) > max_pattern_2_output_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_pattern_2_output_characters_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence_classic, decoded_sentence_argmax 


# sequence target and proba

reverse_output_pattern2_char_index = {i: char 
                            for char, i in pattern_2_output_characters_index.items()}



def sequence_out_proba(pos, lemma_input_seq, target):
    states_value = encoder_model.predict([pos, lemma_input_seq])
    target_seq = np.zeros((1, 1, num_pattern_2_output_characters_tokens))
    target_seq[0, 0, pattern_2_output_characters_index['#']] = 1.
    stop_condition = False
    decoded_sentence = ''
    decode_sequence_proba = ''
    word_target = target[0]
    count = 1
    probability_word = 1

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        letter_target = word_target[count]
        sampled_target_index = pattern_2_output_characters_index[letter_target]
        proba_letter_target = (output_tokens[:, :, sampled_target_index])
        decoded_sentence += (" " + letter_target + ":" + str(proba_letter_target))
        probability_word = probability_word * proba_letter_target
        count = count + 1
        if letter_target == '$': #or
            stop_condition = True
            decode_sequence_proba = str((probability_word))
        target_seq = np.zeros((1, 1, num_pattern_2_output_characters_tokens))
        target_seq[0, 0, sampled_target_index] = 1.
        states_value = [h, c]

    return decode_sequence_proba, decoded_sentence


#####################
# LOAD WUG/DEV FILE #
#####################


data_path = "../data/2023InflectionST/patterns/eng.tst.phono.patterns"
# data_path = "../data/2023InflectionST/patterns/eng.dev.phono.patterns"

poss_dj = []
form_orthos_dj = []
patterns_2_dj_output = []
patterns_2_dj_input = []
patterns_2_dj = []
lemma_phonos_dj = []
poss_type_dj = set()
pattern_2_dj_output_characters = set()
pattern_2_dj_input_characters = set()
lemma_phonos_characters_dj = set()
form_orthos_characters_dj = set()

lines = open(data_path, encoding='utf8').read().split('\n')
for line in lines[: min(len(lines), len(lines) - 1)]:
    line = line.split('\t')
    pos_dj = line[2]
    form_ortho_dj = "#" + line[4] + "$"
    pattern_2_dj = line[6].split('/')
    pattern_2_dj_input = "#" + pattern_2_dj[0] + "$"
    pattern_2_dj_output = "#" + pattern_2_dj[1] + "$"
    lemma_phono_dj = "#" + line[3] + "$"
    poss_dj.append(pos_dj)
    form_orthos_dj.append(form_ortho_dj)
    patterns_2_dj_input.append(pattern_2_dj_input)
    patterns_2_dj_output.append(pattern_2_dj_output)
    lemma_phonos_dj.append(lemma_phono_dj)
    for tipo in poss_dj:
        if tipo not in poss_type_dj:
            poss_type_dj.add(tipo)
    for char in form_ortho_dj:
        if char not in form_orthos_characters_dj:
            form_orthos_characters_dj.add(char) 
    for char in lemma_phono_dj:
        if char not in lemma_phonos_characters_dj:
            lemma_phonos_characters_dj.add(char)             
    for char in pattern_2_dj_input:
        if char not in pattern_2_dj_input_characters:
            pattern_2_dj_input_characters.add(char)             
    for char in pattern_2_dj_output:
        if char not in pattern_2_dj_output_characters:
            pattern_2_dj_output_characters.add(char)  


print(" FILE LOADED")


# encode data in a 3d array 
poss_data_dj = np.zeros((len(poss_dj), len(poss_type)),dtype='float32')
lemma_phonos_data_dj = np.zeros((len(lemma_phonos_dj), max_lemma_phonos_length, num_lemma_phonos_tokens), dtype='float32')
patterns_2_dj_input_data = np.zeros((len(patterns_2_dj_input), max_pattern_2_input_length, num_pattern_2_input_characters_tokens),dtype='float32')
patterns_2_dj_output_data = np.zeros((len(patterns_2_dj_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
decoder_input_data_patterns_2_dj = np.zeros((len(patterns_2_dj_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
decoder_output_data_patterns_2_dj = np.zeros((len(patterns_2_dj_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')

print("INPUT DATA WUG DEV CREATED")


# one-hot encoding

for i, (pattern_2_dj_input, pattern_2_dj_output, lemma_phono_dj) in enumerate(zip(patterns_2_dj_input, patterns_2_dj_output, lemma_phonos_dj)):
    for t, char in enumerate(pattern_2_dj_input):
        patterns_2_dj_input_data[i, t, pattern_2_input_characters_index[char]] = 1.  
    for t, char in enumerate(lemma_phono_dj):
        lemma_phonos_data_dj[i, t, lemma_phonos_characters_index[char]] = 1.          
    for t, char in enumerate(pattern_2_dj_output):
        patterns_2_dj_output_data[i, t, pattern_2_output_characters_index[char]] = 1.      
    for t, char in enumerate(pattern_2_dj_output):
        decoder_input_data_patterns_2_dj[i, t, pattern_2_output_characters_index[char]] = 1.
        if t > 0:
            decoder_output_data_patterns_2_dj[i, t - 1, pattern_2_output_characters_index[char]] = 1.

for i,char in enumerate(poss_dj):
        poss_data_dj[i, poss_type_index[char]] = 1.

print("INPUT DATA VECTORIZATION CREATED")


#######################################
############ PREDICTION WUGS ##########
#######################################

file_PREDICTION_for_analysis = open('D../results/baseline/Calderone/eng.out', 'w', encoding='utf-8')

for seq_index in range(0,len(lemma_phonos_dj)):
    target = patterns_2_dj_output[seq_index: seq_index + 1]
    lemma_input_seq = lemma_phonos_data_dj[seq_index: seq_index + 1]
    pos = poss_data_dj[seq_index: seq_index + 1]
    decoded = sequence_out(pos, lemma_input_seq)
    decoded_proba = sequence_out_proba(pos, lemma_input_seq, target)
    file_PREDICTION_for_analysis.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (lemma_phonos_dj[seq_index], form_orthos_dj[seq_index], poss_dj[seq_index], patterns_2_dj_output[seq_index], decoded, decoded_proba))
file_PREDICTION_for_analysis.close()
print("PREDICTION PROBA DONE!") 



