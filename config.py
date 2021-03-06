MAX_LENGTH = 72

decoding_options_0 = {'max_length': MAX_LENGTH,
                    'repetition_penalty': 1.2,
                    'temperature': 0}

decoding_options_1 = {'do_sample': True,
                      'max_length': MAX_LENGTH,
                      'top_p': 0.98
                     }

decoding_options_2 = {'do_sample':True,
                    'max_length':MAX_LENGTH,
                    'top_k':50,
                    'top_p':0.9}

fallback_decoding = {'max_length': MAX_LENGTH,
                     'num_beams':25,
                     'no_repeat_ngram_size':2,
                     'num_return_sequences': 1,
                     #  'temperature': 0.6,
                     'early_stopping':True}

decoding_options = [decoding_options_0, decoding_options_1, decoding_options_2, fallback_decoding]


