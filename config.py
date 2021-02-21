decoding_options_0 = {'max_length': 96,
                    'repetition_penalty': 1.2,
                    'temperature': 0}

#  decoding_options_1 = {'max_length': 96,
                      #  'num_beams':5,
                      #  'no_repeat_ngram_size':2,
                      #  'early_stopping':True}

decoding_options_1 = {'do_sample': True,
                      'max_length': 96,
                      'top_p': 0.98
                     }

decoding_options_2 = {'do_sample':True,
                    'max_length':96,
                    'top_k':50, 
                    'top_p':0.9}

fallback_decoding = {'max_length': 96,
                     'num_beams':25,
                     'no_repeat_ngram_size':2,
                     'num_return_sequences': 2,
                     'temperature': 0.6,
                     'early_stopping':True}

decoding_options = [decoding_options_0, decoding_options_1, decoding_options_2, fallback_decoding]


