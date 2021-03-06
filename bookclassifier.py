import snownlp
import os
import stat
import pickle
import argparse


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


class FolderDataRW:
    def __init__(self, filename):
        self.filename = filename
    
    def save(self, title_list, folder_list):
        with open(self.filename+'.tflist', 'wb') as handle:
            pickle.dump(title_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(folder_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.filename+'.tflist', 'rb') as handle:
            title_list = pickle.load(handle)
            folder_list = pickle.load(handle)

        return title_list, folder_list

def save_data_sets(filename, X, y, X_test, y_test):
    with open(filename+'.x', 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename+'.y', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename+'.xtest', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename+'.ytest', 'wb') as handle:
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data_sets(filename):
    with open(filename+'.x', 'rb') as handle:
        X = pickle.load(handle)
    with open(filename+'.y', 'rb') as handle:
        y = pickle.load(handle)
    with open(filename+'.xtest', 'rb') as handle:
        X_test = pickle.load(handle)
    with open(filename+'.ytest', 'rb') as handle:
        y_test = pickle.load(handle)

    return X, y, X_test, y_test

def save_vocabulary(vocabulary, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_vocabulary(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)



def seg_chinese_words(sentence):
    s = snownlp.SnowNLP(sentence)
    return " ".join(s.words)


def browse_folder(rootdir, title_list, folder_list ):
    file_count = 0
    for diritem in os.listdir(rootdir):
        pathname = os.path.join(rootdir, diritem)

        try:
            statdata = os.stat(pathname)
        except FileNotFoundError as file_err:
            print(file_err)
            print("Skip file " + pathname)
            continue

        mode = statdata.st_mode
        if stat.S_ISDIR(mode):
            file_type = 'fold'
            print('====', diritem)


            # It's a directory, recurse into it
            browse_folder(pathname, title_list, folder_list)
        elif stat.S_ISREG(mode):
            # It's a file, call the callback function
            dotpos = pathname.rfind('.')
            file_type = ''
            if dotpos > -1:
                file_type = pathname[dotpos+1:].upper()

            title = seg_chinese_words(diritem)

            title_list.append(title)
            folder_list.append(rootdir)
            file_count += 1
        else:
            # Unknown file type, print a message
            print('Skipping %s' % pathname)

    if file_count == 1:
        # training need each target has at least two items. 
        # so we have to duplicate the training data for this case.
        title_list.append(title)
        folder_list.append(rootdir)



def add_to_vocabulary(newdocs, exist_voc):

    # how to add new words into existing vocabulary
    count2 = CountVectorizer(tokenizer=lambda text: text.split(' '))
    count2.fit(newdocs)
    print(count2.vocabulary_)
    print(count2.get_feature_names())

    exist_keys = set(exist_voc.keys())
    new_idx = len(exist_voc)
    for key in count2.vocabulary_.keys():
        if key not in exist_keys:
            exist_voc[key] = new_idx
            new_idx += 1

    print('Words:', len(exist_voc))

class TFGraphBase:
    def __init__(self, model_file):
        self.model_file = model_file

def create_batch_generator(X, y, batch_size=64, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)
    
    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])

def batch_train_tf(sess, X_train, y_train, n_epochs, train_op, cost, saver, model_file):
    ## 50 epochs of training:
    total_training_costs = []
    for epoch in range(n_epochs):
        training_costs = []
        batch_generator = create_batch_generator(X_train, y_train)

        for batch_X, batch_y in batch_generator:
            ## prepare a dict to feed data to our network:
            feed = {"tf_X:0":batch_X, "tf_y:0":batch_y}
            _, batch_cost = sess.run([train_op, cost], feed_dict=feed)

            training_costs.append(batch_cost)

        if epoch % 50 == 0:
            print(' -- Epoch %2d '
                'Avg. Training Loss: %.4f' % (epoch+1, np.mean(training_costs)
                ))

        total_training_costs.append(np.mean(training_costs))

    saver.save(sess, model_file)
    plt.plot(total_training_costs)
    plt.show()


class TFGraphForTraining:
    def __init__(self, model_file):
        self.init_op = None
        self.train_op = None
        self.cost = None
        self.model_file = model_file


    def build_graph(self, feature_count, n_classes):
        g = tf.Graph()
        n_hidden_nodes = 200

        with g.as_default():
            self.tf_X = tf.placeholder(shape=(None, feature_count),
                                dtype=tf.float32,
                                name='tf_X')

            self.tf_y = tf.placeholder(dtype=tf.int32,
                                shape=None, name='tf_y')

            y_onehot = tf.one_hot(indices=self.tf_y, depth=n_classes, name="y_onehot")

            h1 = tf.layers.dense(inputs=self.tf_X, units=n_hidden_nodes,
                                activation=tf.tanh, name='layer1')
            h2 = tf.layers.dense(inputs=h1, units=n_hidden_nodes,
                                activation=tf.tanh, name='layer2')
            logits = tf.layers.dense(inputs=h2,
                                units=n_classes,
                                activation=None,
                                name='logits')
            self.predictions = {
                'classes' : tf.argmax(logits, axis=1, name='predicted_classes'),
                'probabilities' : tf.nn.softmax(logits, name='predict_confidence')
            }


        ## define cost function and optimizer:
        with g.as_default():
            self.cost = tf.losses.softmax_cross_entropy(
                                onehot_labels=y_onehot, logits=logits)
            optimizer = tf.train.GradientDescentOptimizer(
                                learning_rate=0.001)
            self.train_op = optimizer.minimize(loss=self.cost)
            self.init_op = tf.global_variables_initializer()

        # save the trained graph for later usage
        with g.as_default():
            self.saver = tf.train.Saver()

        self.g = g

    def train_tf(self, X_train, y_train, n_epochs):
        ## create a session to launch the graph
        sess = tf.Session(graph=self.g)
        ## run the variable initialization operator
        sess.run(self.init_op)

        batch_train_tf(sess, X_train, y_train, n_epochs, self.train_op, self.cost, self.saver, self.model_file)
        self.sess = sess    

    def verify(self, X_test, y_test):
        ## do prediction on the test set:
        feed = {self.tf_X : X_test}
        y_pred = self.sess.run(self.predictions['classes'],
                                feed_dict=feed)

        print('Test Accuracy: %.2f%%' % (
            100*np.sum(y_pred == y_test)/len(y_test) ))

class TFGraph:
    def __init__(self, model_file):
        self.g2 = tf.Graph()
        self.model_file = model_file
        with self.g2.as_default():
            self.sess = tf.Session(graph=self.g2)
            self.new_saver = tf.train.import_meta_graph(model_file+'.meta')
            self.new_saver.restore(self.sess, model_file)

            self.cost = self.g2.get_collection("losses")[0]
            self.train_op = self.g2.get_collection("train_op")[0]


    def predict(self, x_test):
        y_pred = self.sess.run('predicted_classes:0',
                            feed_dict={'tf_X:0': [x_test]})
        return y_pred

    def confidence(self, x_test):
        return self.sess.run('predict_confidence:0',
                            feed_dict={'tf_X:0': [x_test]})

    def train_tf(self, X_train, y_train, n_epochs):
        batch_train_tf(self.sess, X_train, y_train, n_epochs, 
                self.train_op, self.cost, self.new_saver, self.model_file)


def get_original_words(vector, vocabulary):
    original_words = []
    for idx, item in enumerate(vector):
        if item == 1:
            original_words.append(vocabulary[idx])

    return " ".join(original_words)

def inv_dict(word_to_num):
    num_to_word = dict()
    for word in word_to_num:
        num_to_word[word_to_num[word]] = word
    
    return num_to_word


model_file = './model_file'
vocab_file = "vocab.pickle"
class_map_file = 'classmap.pickle'
data_set_file = "filenamedata"

def split_data(title_list, folder_list):
    count = CountVectorizer(tokenizer=lambda text: text.split(' '))
    docs = np.array(title_list)

    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    print('Words:', len(count.vocabulary_))
    print('Titles:', len(title_list))
    X = bag.toarray()

    save_vocabulary(count.vocabulary_, vocab_file)
    
    class_mapping = {label:idx for idx,label in enumerate(np.unique(folder_list))}
    print(class_mapping)
    save_vocabulary(class_mapping, class_map_file)

    new_folder_list = [class_mapping[item] for item in folder_list]
    print(new_folder_list)

    # ohe = OneHotEncoder()
    # y = ohe.fit_transform(new_folder_list).toarray()
    y = new_folder_list

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.3,
                    random_state=0, stratify=y)
    # pickle four sets
    save_data_sets(data_set_file, X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test, len(class_mapping.keys())

def build_model(title_list, folder_list):

    X_train, X_test, y_train, y_test, n_classes = split_data(title_list, folder_list)

    tfg = TFGraphForTraining(model_file)
    tfg.build_graph(len(X_test[0]), n_classes)
    n_epochs = 50
    tfg.train_tf(X_train, y_train, n_epochs)
    tfg.verify(X_test, y_test)


def get_voc_and_class():
    exist_voc = get_vocabulary(vocab_file)
    print(exist_voc)
    class_mapping = get_vocabulary(class_map_file)
    print(class_mapping)
    folder_mapping = inv_dict(class_mapping)

    return exist_voc, folder_mapping


def classify(work_folder):

    exist_voc, folder_mapping = get_voc_and_class()
    tfgload = TFGraph(model_file)
    count = CountVectorizer(tokenizer=lambda text: text.split(' '), vocabulary=exist_voc)

    old_dir = os.getcwd()
    os.chdir(work_folder)

    cnt = 0
    for diritem in os.listdir('.'):
        try:
            statdata = os.stat(diritem)
        except FileNotFoundError as file_err:
            print(file_err)
            print("Skip file " + diritem)
            continue

        mode = statdata.st_mode
        if stat.S_ISREG(mode):
            # It's a file, call the callback function
            print(diritem)

            newdocs = np.array([seg_chinese_words(diritem)])

            matrix = count.transform(newdocs).toarray()
            pred = tfgload.predict(matrix[0])
            target_folder = folder_mapping[pred[0]]
            print('=====> ', target_folder)
            confidence = tfgload.confidence(matrix[0])[0]
            confidence = sorted(confidence, reverse=True)[:3]
            confidence = [100*val for val in confidence]
            print("Top 3 Confidences(%):", confidence)
            if confidence[0] > 2*confidence[1] or confidence[0] > 6:
                # we only make action when we have enough confidence on the result
                answer = input("Is this right?(any key==yes/[Enter for No]?")
                print('--{a}--'.format(a=answer))
                if len(answer) > 0:
                    # right target.
                    os.renames(diritem, target_folder+'/'+ diritem)
                    # renames() will crate intermediate folder in the target as needed.
                    print(diritem, 'moved.')
            
            print('')
           

    os.chdir(old_dir)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Train TensorFlow model to classify books.')
    #argparser.add_argument("--RAM", dest='RAM_disk', # metavar='Folder-root',
    #                       type=str, default=None, required=True,
    #                       help='root of the RAM disk')

    #for bool value         default=False, required=False, action='store_true',

    argparser.add_argument("--collect", dest='collect',
                           type=str, default=None, required=False, 
                           help='collect folder data rooted in current folder. Specify folder for data storage.')

    argparser.add_argument("--train", dest='train', # metavar='Folder-root',
                           type=str, default=None, required=False, 
                           help='work in training moode, using collected data file located in specified folder')

    argparser.add_argument("--retrain", dest='retrain', # metavar='Folder-root',
                           type=str, default=None, required=False, 
                           help='work in re-training moode, using collected data file located in specified folder')

    argparser.add_argument("--work", dest='work', # metavar='Folder-root',
                           type=str, default=None, required=False, 
                           help='work in work moode, specify the work folder.')

    argparser.add_argument("--merge", dest='merge', # metavar='Folder-root',
                           type=str, default=None, required=False, 
                           help='merge two folder structures, specify the new folder root. Old folder data should be in current directory.')


    args = argparser.parse_args()

    foldedatafilename = 'bcfolderdata'

    if args.merge:
        oldFolderRW = FolderDataRW(foldedatafilename)
        old_title_list, old_folder_list = oldFolderRW.load()

        rootdir = args.merge
        new_title_list = []
        new_folder_list = []
        browse_folder(rootdir, new_title_list, new_folder_list)
        
        # need make values in new_folder_list to be relative path

        start = len(args.merge)+1
        relative_new_folder_list = [item[start:] for item in new_folder_list]

        oldFolderRW.save(old_title_list + new_title_list,
                old_folder_list + relative_new_folder_list)

    elif args.collect:
        rootdir = '.'
        title_list = []
        folder_list = []
        browse_folder(rootdir, title_list, folder_list)

        folderRW = FolderDataRW(args.collect+'/'+foldedatafilename)
        folderRW.save(title_list, folder_list)

    elif args.train:
        folderRW = FolderDataRW(args.train+'/'+foldedatafilename)
        title_list, folder_list = folderRW.load()
        build_model(title_list, folder_list)
    elif args.retrain:

        exist_voc, folder_mapping = get_voc_and_class()
        folder_mapping = inv_dict(folder_mapping)

        folderRW = FolderDataRW(args.retrain+'/'+foldedatafilename)
        title_list, folder_list = folderRW.load()

        newdocs = np.array(title_list)
        add_to_vocabulary(newdocs, exist_voc)

        count3 = CountVectorizer(tokenizer=lambda text: text.split(' '), vocabulary=exist_voc)
        X = count3.transform(newdocs)

        save_vocabulary(count3.vocabulary_, vocab_file)
        
        new_folder_list = [folder_mapping[item] for item in folder_list]

        # ohe = OneHotEncoder()
        # y = ohe.fit_transform(new_folder_list).toarray()
        y = new_folder_list

        X_train, X_test, y_train, y_test =\
            train_test_split(X, y, test_size=0.3,
                        random_state=0, stratify=y)
        # pickle four sets
        save_data_sets(data_set_file, X_train, X_test, y_train, y_test)

        tfgload = TFGraph(model_file)
        n_epochs = len(X_train)
        tfgload.train_tf(X_train, y_train, n_epochs)

    elif args.work is not None:
        classify(args.work)
    else:
        exist_voc, folder_mapping = get_voc_and_class()

        X_train, X_test, y_train, y_test = load_data_sets(data_set_file)
        print("Train size:", len(X_train))
        print("Test size:", len(X_test))



        tfgload = TFGraph(model_file)

        for vector in X_test:
            print(get_original_words(vector, inv_dict(exist_voc)))
            pred = tfgload.predict(vector)
            print('----', folder_mapping[pred[0]])

        new_list = ['ths is a strange title', 'another strange title too 100']
        chinese_list= ['战争理论','欧阳修柳宗元苏轼苏辙选集', '费孝通']
        new_list = [seg_chinese_words(sen) for sen in chinese_list]
        print(new_list)
        newdocs = np.array(new_list)
        print(newdocs)

        add_to_vocabulary(newdocs, exist_voc)

        count3 = CountVectorizer(tokenizer=lambda text: text.split(' '), vocabulary=exist_voc)
        bag2 = count3.transform(newdocs)
        print(bag2.toarray())

        for vector in bag2.toarray():
            print(get_original_words(vector, inv_dict(exist_voc)))
