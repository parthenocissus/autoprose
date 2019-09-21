from keras.callbacks import LearningRateScheduler, Callback
from keras.engine import Layer, InputSpec
from keras.layers import Embedding, SpatialDropout1D, concatenate, Dense, Reshape, Bidirectional, LSTM
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras import backend as K, Input, Model, initializers
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from .autoprose_utilities import *
import csv
import re


class autoprose:

    parameters = {
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 40,
        'max_words': 10000,
        'dim_embeddings': 100,
        'word_level': False,
        'single_text': False
    }
    default_parameters = parameters.copy()
    mtoken = '<s>'

    def __init__(self, weights_path=None,
                 vocab_path=None,
                 parameters_path=None,
                 name="autowriting"):

        if weights_path is None:
            weights_path = resource_filename(__name__,
                                             'default_modelweights.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'default_vocabulary.json')

        if parameters_path is not None:
            with open(parameters_path, 'r',
                      encoding='utf8', errors='ignore') as json_file:
                self.parameters = json.load(json_file)

        self.parameters.update({'name': name})
        self.default_parameters.update({'name': name})

        with open(vocab_path, 'r',
                  encoding='utf8', errors='ignore') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', lower=False, char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = autoprose_model(self.num_classes,
                                      cfg=self.parameters,
                                      weights_path=weights_path)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, prefix=None,
                 temperature=[1.0, 0.5, 0.2, 0.2],
                 max_gen_length=300, interactive=False,
                 top_n=3, progress=True):
        gen_texts = []
        iterable = trange(n) if progress and n > 1 else range(n)
        for _ in iterable:
            gen_text, _ = autoprose_generate(self.model,
                                           self.vocab,
                                           self.indices_char,
                                           temperature,
                                           self.parameters['max_length'],
                                           self.mtoken,
                                           self.parameters['word_level'],
                                           self.parameters.get(
                                               'single_text', False),
                                           max_gen_length,
                                           interactive,
                                           top_n,
                                           prefix)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def generate_samples(self, n=3, temperatures=[0.2, 0.5, 1.0], **kwargs):
        for temperature in temperatures:
            print('#'*20 + '\nTemperature: {}\n'.format(temperature) +
                  '#'*20)
            self.generate(n, temperature=temperature, progress=False, **kwargs)

    def train_on_texts(self, texts, context_labels=None,
                       batch_size=128,
                       num_epochs=50,
                       verbose=1,
                       new_model=False,
                       gen_epochs=1,
                       train_size=1.0,
                       max_gen_length=300,
                       validation=True,
                       dropout=0.0,
                       via_new_model=False,
                       save_epochs=0,
                       multi_gpu=False,
                       **kwargs):

        if new_model and not via_new_model:
            self.train_new_model(texts,
                                 context_labels=context_labels,
                                 num_epochs=num_epochs,
                                 gen_epochs=gen_epochs,
                                 train_size=train_size,
                                 batch_size=batch_size,
                                 dropout=dropout,
                                 validation=validation,
                                 save_epochs=save_epochs,
                                 multi_gpu=multi_gpu,
                                 **kwargs)
            return

        if context_labels:
            context_labels = LabelBinarizer().fit_transform(context_labels)

        # if 'prop_keep' in kwargs:
        #     train_size = prop_keep

        if self.parameters['word_level']:
            texts = [text_to_word_sequence(text, filters='') for text in texts]

        # calculate all combinations of text indices + token indices
        indices_list = [np.meshgrid(np.array(i), np.arange(
            len(text) + 1)) for i, text in enumerate(texts)]
        indices_list = np.block(indices_list)

        # If a single text, there will be 2 extra indices, so remove them
        # Also remove first sequences which use padding
        if self.parameters['single_text']:
            indices_list = indices_list[self.parameters['max_length']:-2, :]

        indices_mask = np.random.rand(indices_list.shape[0]) < train_size

        if multi_gpu:
            num_gpus = len(K.tensorflow_backend._get_available_gpus())
            batch_size = batch_size * num_gpus

        gen_val = None
        val_steps = None
        if train_size < 1.0 and validation:
            indices_list_val = indices_list[~indices_mask, :]
            gen_val = generate_sequences_from_texts(
                texts, indices_list_val, self, context_labels, batch_size)
            val_steps = max(
                int(np.floor(indices_list_val.shape[0] / batch_size)), 1)

        indices_list = indices_list[indices_mask, :]

        num_tokens = indices_list.shape[0]
        assert num_tokens >= batch_size, "Fewer tokens than batch_size."

        level = 'word' if self.parameters['word_level'] else 'character'
        print("Training on {:,} {} sequences.".format(num_tokens, level))

        steps_per_epoch = max(int(np.floor(num_tokens / batch_size)), 1)

        gen = generate_sequences_from_texts(
            texts, indices_list, self, context_labels, batch_size)

        base_lr = 4e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        if context_labels is not None:
            if new_model:
                weights_path = None
            else:
                weights_path = "{}_weights.hdf5".format(self.parameters['name'])
                self.save(weights_path)

            self.model = autoprose_model(self.num_classes,
                                          dropout=dropout,
                                          cfg=self.parameters,
                                          context_size=context_labels.shape[1],
                                          weights_path=weights_path)

        model_t = self.model

        if multi_gpu:
            # Do not locate model/merge on CPU since sample sizes are small.
            parallel_model = multi_gpu_model(self.model,
                                             gpus=num_gpus,
                                             cpu_merge=False)
            parallel_model.compile(loss='categorical_crossentropy',
                                   optimizer=RMSprop(lr=4e-3, rho=0.99))

            model_t = parallel_model
            print("Training on {} GPUs.".format(num_gpus))

        model_t.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                              epochs=num_epochs,
                              callbacks=[
                                  LearningRateScheduler(
                                      lr_linear_decay),
                                  generate_after_epoch(
                                      self, gen_epochs,
                                      max_gen_length),
                                  save_model_weights(
                                      self, num_epochs,
                                      save_epochs)],
                              verbose=verbose,
                              max_queue_size=10,
                              validation_data=gen_val,
                              validation_steps=val_steps
                              )

        # Keep the text-only version of the model if using context labels
        if context_labels is not None:
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])

    def train_new_model(self, texts, context_labels=None, num_epochs=50,
                        gen_epochs=1, batch_size=128, dropout=0.0,
                        train_size=1.0,
                        validation=True, save_epochs=0,
                        multi_gpu=False, **kwargs):
        self.parameters = self.default_parameters.copy()
        self.parameters.update(**kwargs)

        print("Training new model w/ {}-layer, {}-cell {}LSTMs".format(
            self.parameters['rnn_layers'], self.parameters['rnn_size'],
            'Bidirectional ' if self.parameters['rnn_bidirectional'] else ''
        ))

        # If training word level, must add spaces around each punctuation.
        # https://stackoverflow.com/a/3645946/9314418

        if self.parameters['word_level']:
            punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
            for i in range(len(texts)):
                texts[i] = re.sub('([{}])'.format(punct), r' \1 ', texts[i])
                texts[i] = re.sub(' {2,}', ' ', texts[i])

        # Create text vocabulary for new texts
        # if word-level, lowercase; if char-level, uppercase
        self.tokenizer = Tokenizer(filters='',
                                   lower=self.parameters['word_level'],
                                   char_level=(not self.parameters['word_level']))
        self.tokenizer.fit_on_texts(texts)

        # Limit vocab to max_words
        max_words = self.parameters['max_words']
        self.tokenizer.word_index = {k: v for (
            k, v) in self.tokenizer.word_index.items() if v <= max_words}

        if not self.parameters.get('single_text', False):
            self.tokenizer.word_index[self.mtoken] = len(
                self.tokenizer.word_index) + 1
        self.vocab = self.tokenizer.word_index
        self.num_classes = len(self.vocab) + 1
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

        # Create a new, blank model w/ given params
        self.model = autoprose_model(self.num_classes,
                                      dropout=dropout,
                                      cfg=self.parameters)

        # Save the files needed to recreate the model
        with open('{}_vocab.json'.format(self.parameters['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

        with open('{}_parameters.json'.format(self.parameters['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.parameters, outfile, ensure_ascii=False)

        self.train_on_texts(texts, new_model=True,
                            via_new_model=True,
                            context_labels=context_labels,
                            num_epochs=num_epochs,
                            gen_epochs=gen_epochs,
                            train_size=train_size,
                            batch_size=batch_size,
                            dropout=dropout,
                            validation=validation,
                            save_epochs=save_epochs,
                            multi_gpu=multi_gpu,
                            **kwargs)

    def save(self, weights_path="autoprose_weights_saved.hdf5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path):
        self.model = autoprose_model(self.num_classes,
                                      cfg=self.parameters,
                                      weights_path=weights_path)

    def reset(self):
        self.parameters = self.default_parameters.copy()
        self.__init__(name=self.parameters['name'])

    def train_from_file(self, file_path, header=True, delim="\n",
                        new_model=False, context=None,
                        is_csv=False, **kwargs):

        context_labels = None
        if context:
            texts, context_labels = autoprose_texts_from_file_context(
                file_path)
        else:
            texts = autoprose_texts_from_file(file_path, header,
                                               delim, is_csv)

        print("{:,} texts collected.".format(len(texts)))
        if new_model:
            self.train_new_model(
                texts, context_labels=context_labels, **kwargs)
        else:
            self.train_on_texts(texts, context_labels=context_labels, **kwargs)

    def train_on_text(self, file_path, new_model=True, **kwargs):
        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
            texts = [f.read()]

        if new_model:
            self.train_new_model(
                texts, single_text=True, **kwargs)
        else:
            self.train_on_texts(texts, single_text=True, **kwargs)

    def generate_to_file(self, destination_path, **kwargs):
        texts = self.generate(return_as_list=True, **kwargs)
        with open(destination_path, 'w') as f:
            for text in texts:
                f.write("{}\n".format(text))

    def encode_text_vectors(self, texts, pca_dims=50, tsne_dims=None,
                            tsne_seed=None, return_pca=False,
                            return_tsne=False):

        # if a single text, force it into a list:
        if isinstance(texts, str):
            texts = [texts]

        vector_output = Model(inputs=self.model.input,
                              outputs=self.model.get_layer('attention').output)
        encoded_vectors = []
        maxlen = self.parameters['max_length']
        for text in texts:
            if self.parameters['word_level']:
                text = text_to_word_sequence(text, filters='')
            text_aug = [self.mtoken] + list(text[0:maxlen])
            encoded_text = autoprose_encode_sequence(text_aug, self.vocab,
                                                      maxlen)
            encoded_vector = vector_output.predict(encoded_text)
            encoded_vectors.append(encoded_vector)

        encoded_vectors = np.squeeze(np.array(encoded_vectors), axis=1)
        if pca_dims is not None:
            assert len(texts) > 1, "Must use more than 1 text for PCA"
            pca = PCA(pca_dims)
            encoded_vectors = pca.fit_transform(encoded_vectors)

        if tsne_dims is not None:
            tsne = TSNE(tsne_dims, random_state=tsne_seed)
            encoded_vectors = tsne.fit_transform(encoded_vectors)

        return_objects = encoded_vectors
        if return_pca or return_tsne:
            return_objects = [return_objects]
        if return_pca:
            return_objects.append(pca)
        if return_tsne:
            return_objects.append(tsne)

        return return_objects

    def similarity(self, text, texts, use_pca=True):
        text_encoded = self.encode_text_vectors(text, pca_dims=None)
        if use_pca:
            texts_encoded, pca = self.encode_text_vectors(texts,
                                                          return_pca=True)
            text_encoded = pca.transform(text_encoded)
        else:
            texts_encoded = self.encode_text_vectors(texts, pca_dims=None)

        cos_similairity = cosine_similarity(text_encoded, texts_encoded)[0]
        text_sim_pairs = list(zip(texts, cos_similairity))
        text_sim_pairs = sorted(text_sim_pairs, key=lambda x: -x[1])
        return text_sim_pairs


def autoprose_model(num_classes, cfg, context_size=None,
                     weights_path=None,
                     dropout=0.0,
                     optimizer=RMSprop(lr=4e-3, rho=0.99)):
    input = Input(shape=(cfg['max_length'],), name='input')
    embedded = Embedding(num_classes, cfg['dim_embeddings'],
                         input_length=cfg['max_length'],
                         name='embedding')(input)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer = embedded if i is 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn(cfg, i+1)(prev_layer))

    seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
    attention = awa(name='attention')(seq_concat)
    output = Dense(num_classes, name='output', activation='softmax')(attention)

    if context_size is None:
        model = Model(inputs=[input], outputs=[output])
        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    else:
        context_input = Input(
            shape=(context_size,), name='context_input')
        context_reshape = Reshape((context_size,),
                                  name='context_reshape')(context_input)
        merged = concatenate([attention, context_reshape], name='concat')
        main_output = Dense(num_classes, name='context_output',
                            activation='softmax')(merged)

        model = Model(inputs=[input, context_input],
                      outputs=[main_output, output])
        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      loss_weights=[0.8, 0.2])

    return model


def new_rnn(cfg, layer_num):
    use_cudnnlstm = K.backend() == 'tensorflow' and len(K.tensorflow_backend._get_available_gpus()) > 0
    if use_cudnnlstm:
        from keras.layers import CuDNNLSTM
        if cfg['rnn_bidirectional']:
            return Bidirectional(CuDNNLSTM(cfg['rnn_size'],
                                           return_sequences=True),
                                 name='rnn_{}'.format(layer_num))

        return CuDNNLSTM(cfg['rnn_size'],
                         return_sequences=True,
                         name='rnn_{}'.format(layer_num))
    else:
        if cfg['rnn_bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                      return_sequences=True,
                                      recurrent_activation='sigmoid'),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(cfg['rnn_size'],
                    return_sequences=True,
                    recurrent_activation='sigmoid',
                    name='rnn_{}'.format(layer_num))


def generate_sequences_from_texts(texts, indices_list,
                                  autoprose, context_labels,
                                  batch_size=128):
    is_words = autoprose.parameters['word_level']
    is_single = autoprose.parameters['single_text']
    max_length = autoprose.parameters['max_length']
    meta_token = autoprose.mtoken

    if is_words:
        new_tokenizer = Tokenizer(filters='', char_level=True)
        new_tokenizer.word_index = autoprose.vocab
    else:
        new_tokenizer = autoprose.tokenizer

    while True:
        np.random.shuffle(indices_list)

        X_batch = []
        Y_batch = []
        context_batch = []
        count_batch = 0

        for row in range(indices_list.shape[0]):
            text_index = indices_list[row, 0]
            end_index = indices_list[row, 1]

            text = texts[text_index]

            if not is_single:
                text = [meta_token] + list(text) + [meta_token]

            if end_index > max_length:
                x = text[end_index - max_length: end_index + 1]
            else:
                x = text[0: end_index + 1]
            y = text[end_index + 1]

            if y in autoprose.vocab:
                x = process_sequence([x], autoprose, new_tokenizer)
                y = autoprose_encode_cat([y], autoprose.vocab)

                X_batch.append(x)
                Y_batch.append(y)

                if context_labels is not None:
                    context_batch.append(context_labels[text_index])

                count_batch += 1

                if count_batch % batch_size == 0:
                    X_batch = np.squeeze(np.array(X_batch))
                    Y_batch = np.squeeze(np.array(Y_batch))
                    context_batch = np.squeeze(np.array(context_batch))

                    # print(X_batch.shape)

                    if context_labels is not None:
                        yield ([X_batch, context_batch], [Y_batch, Y_batch])
                    else:
                        yield (X_batch, Y_batch)
                    X_batch = []
                    Y_batch = []
                    context_batch = []
                    count_batch = 0


def process_sequence(X, autoprose, new_tokenizer):
    X = new_tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(
        X, maxlen=autoprose.parameters['max_length'])

    return X


class awa(Layer):

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(awa, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(awa, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0],
                                                   input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None