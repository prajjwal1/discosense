token_max_length = 75

#  decoding_options_0 = {
    #  "repetition_penalty": 1.2,
    #  "temperature": 0.2,
    #  "no_repeat_ngram_size": 2,
    #  "length_penalty": 0.8,
#  }

#  decoding_options_1 = {
    #  "do_sample": True,
    #  "temperature": 0.4,
    #  "top_k": 40,
    #  "top_p": 0.9,
    #  "no_repeat_ngram_size": 2,
    #  "repetition_penalty": 1.1,
    #  "length_penalty": 0.9,
#  }

decoding_options = {
    "do_sample": True,
    "top_p": 0.98,
    #  "temperature": 0.88,
    "no_repeat_ngram_size": 2,
    "length_penalty": 0.8,
}

#  decoding_af = {
    #  "do_sample": True,
    #  "top_p": 0.98,
    #  "temperature": 0.7,
    #  #  "repetition_penalty": 1.2,
    #  "no_length_ngram_size": 2,
    #  "length_penalty": 0.6
#  }

#  fallback_decoding = {
#  "num_beams": 25,
#  "no_repeat_ngram_size": 3,
#  "num_return_sequences": 1,
#  "early_stopping": True,
#  }

decoding_options = [
    decoding_options
    #  decoding_options_1,
    #  decoding_options_2,
    #  decoding_af
]
