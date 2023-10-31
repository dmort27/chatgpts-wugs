from __future__ import print_function
import os
import random
import argparse

import tensorflow as tf
from tensorflow.keras.models import (
    Model,
    Sequential,
    load_model
)
from tensorflow.keras.layers import (
    Input,
    LSTM, Dense,
    Dropout,
    Concatenate,
    RepeatVector,
    Activation,
    Flatten,
    Masking,
    Bidirectional,
    TimeDistributed
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    ReduceLROnPlateau,
    EarlyStopping
)

import numpy as np
import pandas as pd
import pickle


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
if __name__ == '__main__':
    
    print("This process has the PID: ", os.getpid())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='eng', choices=['eng', 'deu', 'tur', 'tam'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early_stopping', action='store_true')
    args = parser.parse_args()
    
    seed = args.seed
    lang = args.lang
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    early_stopping = args.early_stopping
    size_LSTM = 200
    
    print("\nArguments: \n\n")
    for key, val in vars(args).items():
        print(f"{key}: {val}")
    print("\n\n")
    set_random_seed(seed)

    print("\n\nUsing language: ", lang, "\n\n")

    """----------------------------------------------- Merged Data -----------------------------------------------"""

    ##### load merging file to set the vectors length #####

    # Use the merged_with_nonce.phono.patterns if you have the nonce data
    # data_path = f"../data/{lang}/patterns/{lang}.merged_with_nonce.phono.patterns"
    data_path = f"../data/{lang}/patterns/{lang}.merged.phono.patterns"

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
        
        # lemma_phono_m = "#" + line[0] + "$"
        lemma_phono_m = "#" + line[5] + "$"
        
        # pos_m = line[2]
        pos_m = line[7]
        
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
                             
    print(len(poss_m), len(lemma_phonos_m), len(patterns_2_m_input), len(patterns_2_m_output), len(form_orthos_m))
    for i in range(20):
        print(f"lemma: {lemma_phonos_m[i]}\tldecoder input: {patterns_2_m_input[i]}\tFAP2: {patterns_2_m_output[i]}")
        
    print("\n\nMERGING FILE LOADED\n\n")
    
    """----------------------------------------------- Training Data -----------------------------------------------"""

    ##### load merging file to set the vectors length #####

    data_path = f"../data/{lang}/patterns/{lang}.trn.phono.patterns"

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
        
        # lemma_phono = "#" + line[0] + "$"
        lemma_phono = "#" + line[5] + "$"
        
    #     pos = line[2]
        pos = line[7]
        
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

    print(len(poss), len(lemma_phonos), len(patterns_2_input), len(patterns_2_output), len(form_orthos))
    for i in range(20):
        print(f"lemma: {lemma_phonos[i]}\tldecoder input: {patterns_2_input[i]}\tFAP2: {patterns_2_output[i]}")
        
    print("\n\nTRAIN FILE LOADED\n\n")

    pattern_2_input_characters = sorted(list(pattern_2_m_input_characters_m))
    pattern_2_output_characters = sorted(list(pattern_2_m_output_characters_m))
    form_orthos_characters = sorted(list(form_orthos_characters_m))
    lemma_phonos_characters = sorted(list(lemma_phonos_characters_m))
    poss_type = sorted(list(poss_type_m))

    print("\n\nCHARACTERS =  MERGE CHARACTERS")

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

    print("\n\nINPUT TRAIN DATA CREATED")

    print(poss_data.shape)
    print(lemma_phonos_data.shape)
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

    print("\n\nINPUT TRAIN DATA VECTORIZATION CREATED")


    """----------------------------------------------- Dev Data -----------------------------------------------"""

    #####################
    # LOAD DEV FILE #
    #####################


    data_path =  f"../data/{lang}/patterns/{lang}.dev.phono.patterns"

    poss_dev = []
    form_orthos_dev = []
    patterns_2_dev_output = []
    patterns_2_dev_input = []
    patterns_2_dev = []
    lemma_phonos_dev = []
    poss_type_dev = set()
    pattern_2_dev_output_characters = set()
    pattern_2_dev_input_characters = set()
    lemma_phonos_characters_dev = set()
    form_orthos_characters_dev = set()

    lines = open(data_path, encoding='utf8').read().split('\n')
    for line in lines[: min(len(lines), len(lines) - 1)]:
        line = line.split('\t')
        
        # form_ortho_dev = "#" + line[4] + "$"
        form_ortho_dev = "#" + line[6] + "$"
        
    #     pos_dev = line[2]
        pos_dev = line[7]
        
        # pattern_2_dev = line[6].split('/')
        pattern_2_dev = line[8].split('/')
        
        pattern_2_dev_input = "#" + pattern_2_dev[0] + "$"
        pattern_2_dev_output = "#" + pattern_2_dev[1] + "$"
        
        # lemma_phono_dev = "#" + line[3] + "$"
        lemma_phono_dev = "#" + line[5] + "$"
        
        poss_dev.append(pos_dev)
        form_orthos_dev.append(form_ortho_dev)
        patterns_2_dev_input.append(pattern_2_dev_input)
        patterns_2_dev_output.append(pattern_2_dev_output)
        lemma_phonos_dev.append(lemma_phono_dev)
        for tipo in poss_dev:
            if tipo not in poss_type_dev:
                poss_type_dev.add(tipo)
        for char in form_ortho_dev:
            if char not in form_orthos_characters_dev:
                form_orthos_characters_dev.add(char)
        for char in lemma_phono_dev:
            if char not in lemma_phonos_characters_dev:
                lemma_phonos_characters_dev.add(char)
        for char in pattern_2_dev_input:
            if char not in pattern_2_dev_input_characters:
                pattern_2_dev_input_characters.add(char)
        for char in pattern_2_dev_output:
            if char not in pattern_2_dev_output_characters:
                pattern_2_dev_output_characters.add(char)


    print(len(poss_dev), len(lemma_phonos_dev), len(patterns_2_dev_input), len(patterns_2_dev_output), len(form_orthos_dev))
    for i in range(20):
        print(f"lemma: {lemma_phonos_dev[i]}\tldecoder input: {patterns_2_dev_input[i]}\tFAP2: {patterns_2_dev_output[i]}")

    print("\n\nDEV FILE LOADED\n\n")

    # encode data in a 3d array
    poss_data_dev = np.zeros((len(poss_dev), len(poss_type)),dtype='float32')
    lemma_phonos_data_dev = np.zeros((len(lemma_phonos_dev), max_lemma_phonos_length, num_lemma_phonos_tokens), dtype='float32')
    patterns_2_dev_input_data = np.zeros((len(patterns_2_dev_input), max_pattern_2_input_length, num_pattern_2_input_characters_tokens),dtype='float32')
    patterns_2_dev_output_data = np.zeros((len(patterns_2_dev_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
    decoder_input_data_patterns_2_dev = np.zeros((len(patterns_2_dev_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
    decoder_output_data_patterns_2_dev = np.zeros((len(patterns_2_dev_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')

    print("\n\nINPUT DEV DATA CREATED")

    print(poss_data_dev.shape)
    print(lemma_phonos_data_dev.shape)
    print(patterns_2_dev_input_data.shape)
    print(patterns_2_dev_output_data.shape)
    print(decoder_input_data_patterns_2_dev.shape)
    print(decoder_output_data_patterns_2_dev.shape)

    # one-hot encoding

    for i, (pattern_2_dev_input, pattern_2_dev_output, lemma_phono_dev) in enumerate(zip(patterns_2_dev_input, patterns_2_dev_output, lemma_phonos_dev)):
        for t, char in enumerate(pattern_2_dev_input):
            # if lang == "tur" and char == "z":
            #     continue
            patterns_2_dev_input_data[i, t, pattern_2_input_characters_index[char]] = 1.
        for t, char in enumerate(lemma_phono_dev):
            lemma_phonos_data_dev[i, t, lemma_phonos_characters_index[char]] = 1.
        for t, char in enumerate(pattern_2_dev_output):
            patterns_2_dev_output_data[i, t, pattern_2_output_characters_index[char]] = 1.
        for t, char in enumerate(pattern_2_dev_output):
            decoder_input_data_patterns_2_dev[i, t, pattern_2_output_characters_index[char]] = 1.
            if t > 0:
                decoder_output_data_patterns_2_dev[i, t - 1, pattern_2_output_characters_index[char]] = 1.

    for i,char in enumerate(poss_dev):
            poss_data_dev[i, poss_type_index[char]] = 1.

    print("\n\nINPUT DEV DATA VECTORIZATION CREATED")


    """----------------------------------------------- Model Training -----------------------------------------------"""

    #######################################
    # NETWORK CONFIGURATION BIDIRECTIONAL #
    #######################################
    from keras.layers import TimeDistributed
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

    print("\n\nMODEL CREATED")

    ckpt = f"../models/{lang}/sigmorpho_MODEL2_pattern2_{lang}_lr_{lr}_bs_{batch_size}_seed_{seed}_best.h5"
    # model_checkpoint_callback = ModelCheckpoint(
    #     filepath=ckpt,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='auto',
    #     save_freq = 'epoch',
    #     save_best_only=True
    # )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss', 
        mode='auto',
        factor=0.2, 
        patience=5, 
        min_lr=0.0005, 
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=10,
        verbose=1
    )

    print(f"\n\nModel architecture: \n{model.summary()}")
    # model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    if early_stopping:
        history = model.fit(
            [poss_data, lemma_phonos_data, decoder_input_data_patterns_2], 
            decoder_output_data_patterns_2,
            validation_data=[[poss_data_dev, lemma_phonos_data_dev, decoder_input_data_patterns_2_dev], decoder_output_data_patterns_2_dev],
            batch_size=batch_size, 
            epochs=epochs,
            callbacks=[reduce_lr_callback, early_stopping_callback]
        )
    else:
        history = model.fit(
            [poss_data, lemma_phonos_data, decoder_input_data_patterns_2], 
            decoder_output_data_patterns_2,
            validation_data=[[poss_data_dev, lemma_phonos_data_dev, decoder_input_data_patterns_2_dev], decoder_output_data_patterns_2_dev],
            batch_size=batch_size, 
            epochs=epochs,
            callbacks=[reduce_lr_callback]
        )
    # model.load_weights(ckpt)

    """----------------------------------------------- Model Inference -----------------------------------------------"""

    ###########################################
    # INFERENCE ENCODER/DECODER BIDIRECTIONAL #
    ###########################################
    # encoder

    encoder_model = Model(inputs = [input_POS, encoder_lemma], outputs = encoder_states_ALL)
    print("\n\nINFERENCE ENCODER LOADED\n\n")

    # decoder

    decoder_state_H = Input(shape=(size_LSTM*2,))
    decoder_state_C = Input(shape=(size_LSTM*2,))
    decoder_states_inputs = [decoder_state_H, decoder_state_C]
    decoder_outputs, state_H, state_C = decoder_LSTM(decoder_pattern2, initial_state=decoder_states_inputs)
    decoder_states = [state_H, state_C]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_pattern2] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    print("\n\nINFERENCE DECODER LOADED\n\n")

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


    #######################################
    ############ PREDICTION DEV ##########
    #######################################

    print("\n\nSAVING DEV PREDICTIONS: \n\n")
    file_PREDICTION_for_analysis = open(f'../results/{lang}/{lang}.dev.phono.patterns_PROBA_lr_{lr}_bs_{batch_size}_epochs_{epochs}_early_stopping_{early_stopping}_seed_{seed}', 'w', encoding='utf-8')

    for seq_index in range(0,len(lemma_phonos_dev)):
        target = patterns_2_dev_output[seq_index: seq_index + 1]
        lemma_input_seq = lemma_phonos_data_dev[seq_index: seq_index + 1]
        pos = poss_data_dev[seq_index: seq_index + 1]
        decoded = sequence_out(pos, lemma_input_seq)
        decoded_proba = sequence_out_proba(pos, lemma_input_seq, target)
        file_PREDICTION_for_analysis.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (lemma_phonos_dev[seq_index], form_orthos_dev[seq_index], poss_dev[seq_index], patterns_2_dev_output[seq_index], decoded, decoded_proba))
    file_PREDICTION_for_analysis.close()

    print("\n\nDEV PREDICTION PROBA DONE!\n\n")
        
        
    """----------------------------------------------- Test Data -----------------------------------------------"""

    #####################
    # LOAD TEST FILE #
    #####################
        
    if lang != "tam":        
        data_path =  f"../data/{lang}/patterns/{lang}.tst.phono.patterns"
        
        poss_test = []
        form_orthos_test = []
        patterns_2_test_output = []
        patterns_2_test_input = []
        patterns_2_test = []
        lemma_phonos_test = []
        poss_type_test = set()
        pattern_2_test_output_characters = set()
        pattern_2_test_input_characters = set()
        lemma_phonos_characters_test = set()
        form_orthos_characters_test = set()

        lines = open(data_path, encoding='utf8').read().split('\n')
        for line in lines[: min(len(lines), len(lines) - 1)]:
            line = line.split('\t')

            # form_ortho_test = "#" + line[4] + "$"
            form_ortho_test = "#" + line[6] + "$"

        #     pos_test = line[2]
            pos_test = line[7]

            # pattern_2_test = line[6].split('/')
            pattern_2_test = line[8].split('/')

            pattern_2_test_input = "#" + pattern_2_test[0] + "$"
            pattern_2_test_output = "#" + pattern_2_test[1] + "$"

            # lemma_phono_test = "#" + line[3] + "$"
            lemma_phono_test = "#" + line[5] + "$"

            poss_test.append(pos_test)
            form_orthos_test.append(form_ortho_test)
            patterns_2_test_input.append(pattern_2_test_input)
            patterns_2_test_output.append(pattern_2_test_output)
            lemma_phonos_test.append(lemma_phono_test)
            for tipo in poss_test:
                if tipo not in poss_type_test:
                    poss_type_test.add(tipo)
            for char in form_ortho_test:
                if char not in form_orthos_characters_test:
                    form_orthos_characters_test.add(char)
            for char in lemma_phono_test:
                if char not in lemma_phonos_characters_test:
                    lemma_phonos_characters_test.add(char)
            for char in pattern_2_test_input:
                if char not in pattern_2_test_input_characters:
                    pattern_2_test_input_characters.add(char)
            for char in pattern_2_test_output:
                if char not in pattern_2_test_output_characters:
                    pattern_2_test_output_characters.add(char)


        print(len(poss_test), len(lemma_phonos_test), len(patterns_2_test_input), len(patterns_2_test_output), len(form_orthos_test))
        for i in range(20):
            print(f"lemma: {lemma_phonos_test[i]}\tldecoder input: {patterns_2_test_input[i]}\tFAP2: {patterns_2_test_output[i]}")

        print("\n\nTEST FILE LOADED\n\n")

        # encode data in a 3d array
        poss_data_test = np.zeros((len(poss_test), len(poss_type)),dtype='float32')
        lemma_phonos_data_test = np.zeros((len(lemma_phonos_test), max_lemma_phonos_length, num_lemma_phonos_tokens), dtype='float32')
        patterns_2_test_input_data = np.zeros((len(patterns_2_test_input), max_pattern_2_input_length, num_pattern_2_input_characters_tokens),dtype='float32')
        patterns_2_test_output_data = np.zeros((len(patterns_2_test_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
        decoder_input_data_patterns_2_test = np.zeros((len(patterns_2_test_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
        decoder_output_data_patterns_2_test = np.zeros((len(patterns_2_test_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')

        print("\n\nINPUT TEST DATA CREATED")

        print(poss_data_test.shape)
        print(lemma_phonos_data_test.shape)
        print(patterns_2_test_input_data.shape)
        print(patterns_2_test_output_data.shape)
        print(decoder_input_data_patterns_2_test.shape)
        print(decoder_output_data_patterns_2_test.shape)

        # one-hot encoding

        for i, (pattern_2_test_input, pattern_2_test_output, lemma_phono_test) in enumerate(zip(patterns_2_test_input, patterns_2_test_output, lemma_phonos_test)):
            for t, char in enumerate(lemma_phono_test):
                lemma_phonos_data_test[i, t, lemma_phonos_characters_index[char]] = 1.

        for i,char in enumerate(poss_test):
                poss_data_test[i, poss_type_index[char]] = 1.

        print("\n\nINPUT TEST DATA VECTORIZATION CREATED")

        #######################################
        ############ PREDICTION TEST ##########
        #######################################
        
        print("\n\nSAVING TEST PREDICTIONS: \n\n")
        
        file_PREDICTION_for_analysis = open(f'../results/{lang}/{lang}.tst.phono.patterns_PROBA_lr_{lr}_bs_{batch_size}_epochs_{epochs}_early_stopping_{early_stopping}_seed_{seed}', 'w', encoding='utf-8')
        
        for seq_index in range(0,len(lemma_phonos_test)):
            target = patterns_2_test_output[seq_index: seq_index + 1]
            lemma_input_seq = lemma_phonos_data_test[seq_index: seq_index + 1]
            pos = poss_data_test[seq_index: seq_index + 1]
            decoded = sequence_out(pos, lemma_input_seq)
            decoded_proba = sequence_out_proba(pos, lemma_input_seq, target)
            file_PREDICTION_for_analysis.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (lemma_phonos_test[seq_index], form_orthos_test[seq_index], poss_test[seq_index], patterns_2_test_output[seq_index], decoded, decoded_proba))
        file_PREDICTION_for_analysis.close()

        print("\n\nTEST PREDICTION PROBA DONE!\n\n")
        
    """----------------------------------------------- Wug Data -----------------------------------------------"""
    
    #####################
    # LOAD WUG FILE #
    #####################

    data_path =  f"../data/{lang}/patterns/{lang}.nonce.phono.patterns"

    poss_nonce = []
    form_orthos_nonce = []
    patterns_2_nonce_output = []
    patterns_2_nonce_input = []
    patterns_2_nonce = []
    lemma_phonos_nonce = []
    poss_type_nonce = set()
    pattern_2_nonce_output_characters = set()
    pattern_2_nonce_input_characters = set()
    lemma_phonos_characters_nonce = set()
    form_orthos_characters_nonce = set()

    lines = open(data_path, encoding='utf8').read().split('\n')
    for line in lines[: min(len(lines), len(lines) - 1)]:
        line = line.split('\t')

        # form_ortho_nonce = "#" + line[4] + "$"
        form_ortho_nonce = "#" + line[6] + "$"

    #     pos_nonce = line[2]
        pos_nonce = line[7]

        # pattern_2_nonce = line[6].split('/')
        pattern_2_nonce = line[8].split('/')

        pattern_2_nonce_input = "#" + pattern_2_nonce[0] + "$"
        pattern_2_nonce_output = "#" + pattern_2_nonce[1] + "$"

        # lemma_phono_nonce = "#" + line[3] + "$"
        lemma_phono_nonce = "#" + line[5] + "$"

        poss_nonce.append(pos_nonce)
        form_orthos_nonce.append(form_ortho_nonce)
        patterns_2_nonce_input.append(pattern_2_nonce_input)
        patterns_2_nonce_output.append(pattern_2_nonce_output)
        lemma_phonos_nonce.append(lemma_phono_nonce)
        for tipo in poss_nonce:
            if tipo not in poss_type_nonce:
                poss_type_nonce.add(tipo)
        for char in form_ortho_nonce:
            if char not in form_orthos_characters_nonce:
                form_orthos_characters_nonce.add(char)
        for char in lemma_phono_nonce:
            if char not in lemma_phonos_characters_nonce:
                lemma_phonos_characters_nonce.add(char)
        for char in pattern_2_nonce_input:
            if char not in pattern_2_nonce_input_characters:
                pattern_2_nonce_input_characters.add(char)
        for char in pattern_2_nonce_output:
            if char not in pattern_2_nonce_output_characters:
                pattern_2_nonce_output_characters.add(char)


    print(len(poss_nonce), len(lemma_phonos_nonce), len(patterns_2_nonce_input), len(patterns_2_nonce_output), len(form_orthos_nonce))
    for i in range(20):
        print(f"lemma: {lemma_phonos_nonce[i]}\tldecoder input: {patterns_2_nonce_input[i]}\tFAP2: {patterns_2_nonce_output[i]}")

    print("\n\nWug FILE LOADED\n\n")

    # encode data in a 3d array
    poss_data_nonce = np.zeros((len(poss_nonce), len(poss_type)),dtype='float32')
    lemma_phonos_data_nonce = np.zeros((len(lemma_phonos_nonce), max_lemma_phonos_length, num_lemma_phonos_tokens), dtype='float32')
    patterns_2_nonce_input_data = np.zeros((len(patterns_2_nonce_input), max_pattern_2_input_length, num_pattern_2_input_characters_tokens),dtype='float32')
    patterns_2_nonce_output_data = np.zeros((len(patterns_2_nonce_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
    decoder_input_data_patterns_2_nonce = np.zeros((len(patterns_2_nonce_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')
    decoder_output_data_patterns_2_nonce = np.zeros((len(patterns_2_nonce_output), max_pattern_2_output_length, num_pattern_2_output_characters_tokens),dtype='float32')

    print("\n\nINPUT Wug DATA CREATED")

    print(poss_data_nonce.shape)
    print(lemma_phonos_data_nonce.shape)
    print(patterns_2_nonce_input_data.shape)
    print(patterns_2_nonce_output_data.shape)
    print(decoder_input_data_patterns_2_nonce.shape)
    print(decoder_output_data_patterns_2_nonce.shape)

    # one-hot encoding

    for i, (pattern_2_nonce_input, pattern_2_nonce_output, lemma_phono_nonce) in enumerate(zip(patterns_2_nonce_input, patterns_2_nonce_output, lemma_phonos_nonce)):
        for t, char in enumerate(lemma_phono_nonce):
            lemma_phonos_data_nonce[i, t, lemma_phonos_characters_index[char]] = 1.

    for i,char in enumerate(poss_nonce):
            poss_data_nonce[i, poss_type_index[char]] = 1.

    print("\n\nINPUT WUG DATA VECTORIZATION CREATED")


    """----------------------------------------------- Predictions -----------------------------------------------"""

    #######################################
    ############ PREDICTION WUG ##########
    #######################################

    print("\n\nSAVING WUG PREDICTIONS: \n\n")
    
    file_PREDICTION_for_analysis = open(f'../results/{lang}/{lang}.nonce.phono.patterns_PROBA_lr_{lr}_bs_{batch_size}_epochs_{epochs}_early_stopping_{early_stopping}_seed_{seed}', 'w', encoding='utf-8')

    for seq_index in range(0,len(lemma_phonos_nonce)):
        target = patterns_2_nonce_output[seq_index: seq_index + 1]
        lemma_input_seq = lemma_phonos_data_nonce[seq_index: seq_index + 1]
        pos = poss_data_nonce[seq_index: seq_index + 1]
        decoded = sequence_out(pos, lemma_input_seq)
        decoded_proba = sequence_out_proba(pos, lemma_input_seq, target)
        file_PREDICTION_for_analysis.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (lemma_phonos_nonce[seq_index], form_orthos_nonce[seq_index], poss_nonce[seq_index], patterns_2_nonce_output[seq_index], decoded, decoded_proba))
    file_PREDICTION_for_analysis.close()

    print("\n\nWUG PREDICTION PROBA DONE!\n\n")
