import matplotlib.pyplot as plt
import numpy as np
import string
import seaborn as sns


def read_data(train=True):
    letter_to_num = dict(zip(string.ascii_lowercase, range(26)))
    if train:
        file_obj = open('letters.train.data')
    else:
        file_obj = open('letters.test.data')
    letter_ids = []
    letters = []
    next_ids = []
    images = []
    word_ids = []
    positions = []
    for line in file_obj:
        line_split = line.split()

        letter_ids.append(int(line_split[0].strip()))
        letters.append(letter_to_num[line_split[1].strip()])
        next_ids.append(int(line_split[2].strip()))
        word_ids.append(int(line_split[3].strip()))
        positions.append(int(line_split[4].strip()))
        images.append([int(char) for char in line_split[-128:]])

    letter_ids = np.asarray(letter_ids)
    letters = np.asarray(letters)
    next_ids = np.asarray(next_ids)
    word_ids = np.asarray(word_ids)
    positions = np.asarray(positions)
    images = np.asarray(images)

    if train:
        np.save('letter_ids', letter_ids)
        np.save('letters', letters)
        np.save('next_ids', next_ids)
        np.save('word_ids', word_ids)
        np.save('positions', positions)
        np.save('images', images)
    else:
        np.save('test_ids', letter_ids)
        np.save('test_letters', letters)
        np.save('test_next_ids', next_ids)
        np.save('test_word_ids', word_ids)
        np.save('test_positions', positions)
        np.save('test_images', images)
    file_obj.close()


def train_multiclass_perceptron(letters, images):
    num_examples = len(letters)
    w_dim = (26, 128)
    W = np.zeros(w_dim)
    all_w = []
    epochs = 10
    for e in range(epochs):
        randomize = np.arange(num_examples)
        np.random.shuffle(randomize)
        letters_shuf = letters[randomize]
        images_shuf = images[randomize]
        for let, img in zip(letters_shuf, images_shuf):
            pred_let = np.argmax(np.dot(W, img))
            if pred_let != let:
                W[let, :] = np.add(W[let, :], img)
                W[pred_let, :] = np.subtract(W[pred_let, :], img)
            all_w.append(W)
    return np.mean(all_w, axis=0)


def test_multiclass_perceptron(W, test_y, test_x):
    num_correct = 0
    all_pred = []
    for img, let in zip(test_x, test_y):
        pred = np.argmax(np.dot(W, img))
        all_pred.append(pred)
        num_correct += (let == pred)
    acc = np.floor(num_correct * 1000 / len(test_y)) / 10
    name = 'model1 {}%'.format(acc)
    np.save(name, all_pred)
    return acc


def phi_xy1(x, y):
    phi = np.zeros(phi_length1)
    phi[y*feature_length:(y+1)*feature_length] = x
    return phi


def train_structured1(letters, images):
    num_examples = len(letters)
    correct = 0
    W = np.zeros(phi_length1)
    all_w = []
    epochs = 5
    for e in range(epochs):
        randomize = np.arange(num_examples)
        np.random.shuffle(randomize)
        letters_shuf = letters[randomize]
        images_shuf = images[randomize]
        for let, img in zip(letters_shuf, images_shuf):
            pred = np.argmax([np.dot(W, phi_xy1(img, poss_y)) for poss_y in range(26)])
            if pred != let:
                W = np.add(np.subtract(W, phi_xy1(img, pred)), phi_xy1(img, let))
            else:
                correct += 1
            all_w.append(W)
        print(correct/num_examples)
        correct = 0
    return np.mean(all_w, axis=0)


def test_structured1(W, test_y, test_x,):
    all_pred = []
    num_correct = 0
    for img, let in zip(test_x, test_y):
        pred = np.argmax([np.dot(W, phi_xy1(img, poss_y)) for poss_y in range(26)])
        all_pred.append(pred)
        num_correct += (let == pred)
    acc = np.floor(num_correct * 1000 / len(test_y)) / 10
    name = 'model2 {}%'.format(acc)
    np.save(name, all_pred)
    return acc


def phi_xy2(x, prev_y, curr_y):
    phi = np.zeros(phi_length2)
    phi[curr_y*feature_length:(curr_y+1)*feature_length] = x
    bi_gram_index = prev_y*26 + curr_y  # prev - 0-26, curr - 0-25. i - 0-701 (26*26+25)
    phi[phi_length1 + bi_gram_index] = 1  # phi_length1 denotes the 26*128 first elements of the vector
    return phi


def train_structured2(letters, images, words_ids):
    W = np.zeros(phi_length2)
    # W = np.load('W.npy')
    all_w = []
    all_w_means = []
    diff_w_indx = np.unique(words_ids)
    indexes_list = []
    for dx in diff_w_indx:
        letter_indexes = np.where(words_ids == dx)
        indexes_list.append(letter_indexes)
    num_examples = len(diff_w_indx)
    for _ in range(4):
        correct = 0
        wrong = 0
        shuff_indexes = np.arange(num_examples)
        np.random.shuffle(shuff_indexes)
        for i in shuff_indexes:
            letters_word = letters[indexes_list[i]]
            images_word = images[indexes_list[i]]
            y_hat = structured_prediction2(letters_word, images_word, W)
            if letters_word[0] != y_hat[0]:
                W = np.subtract(np.add(W, phi_xy2(images_word[0], 26, letters_word[0])),
                                phi_xy2(images_word[0], 26, y_hat[0]))
                wrong += 1
            else:
                correct += 1
            for i in range(1, len(letters_word)):
                if letters_word[i] != y_hat[i]:
                    W = np.subtract(np.add(W, phi_xy2(images_word[i], letters_word[i - 1], letters_word[i])),
                                    phi_xy2(images_word[i], letters_word[i - 1], y_hat[i]))
                    wrong += 1
                else:
                    correct += 1

            all_w.append(W)
        print(correct / (correct + wrong))
        mean_w = np.mean(all_w, axis=0)
        W = mean_w
        all_w_means.append(mean_w)
        all_w = []
    return np.mean(all_w_means, axis=0)


def structured_prediction2(letters, images, W):
    n_letters = len(letters)
    D_scores = np.zeros((n_letters, 27))
    D_prev_char = np.zeros((n_letters, 27), dtype=int)
    # The index of the "word beginning" char is 26. (the 27'th letter)
    for i in range(26):
        phi = phi_xy2(images[0], 26, i)
        s = np.dot(W, phi)
        D_scores[0,i] = s
        D_prev_char[0,i] = 26
    for i in range(1, n_letters):
        for j in range(26):
            possib = [np.dot(W, phi_xy2(images[i], prev_y, j)) + D_scores[i-1, prev_y] for prev_y in range(26)]
            best_i = np.argmax(possib)
            best_s = possib[best_i]
            D_scores[i,j] = best_s
            D_prev_char[i,j] = best_i
    y_hat = np.zeros(n_letters, dtype=int)
    y_hat[-1] = np.argmax([D_scores[n_letters - 1, i] for i in range(26)])

    for i in range(n_letters - 2, -1, -1):
        y_hat[i] = D_prev_char[i+1, int(y_hat[i+1])]

    return y_hat


def test_structured2(W, letters, images, words_ids):
    all_pred = []
    correct = 0
    prev_w_id = words_ids[0]
    letters_indexes = []
    for idx, w_id in enumerate(words_ids):
        if w_id > prev_w_id:
            letters_word = letters[letters_indexes]
            images_word = images[letters_indexes]
            y_hat = structured_prediction2(letters_word, images_word, W)

            for i in range(len(letters_word)):
                correct += (letters_word[i] == y_hat[i])
                all_pred.append(letters_word[i])
            letters_indexes = []
            prev_w_id = w_id
        if w_id == prev_w_id:
            letters_indexes.append(idx)
    acc = np.floor((correct * 1000) / len(letters)) / 10
    name = 'Accuracy {}%'.format(acc)
    print(name)
    np.save(name, all_pred)
    return acc


def draw_heat_map(w):
    # the 27*26 last numbers are the relevant values of the heat_map
    relev_w = w[-26*27:]
    # bi_gram_index = prev_y * 26 + curr_y
    # curr_y = np.mod(index, 26)
    # prev_y = np.floor(index/27)
    resh = np.reshape(relev_w, (27, 26))

    plt.figure()
    ax = sns.heatmap(resh, linewidth=0.5)
    x_abc = list(string.ascii_lowercase)
    y_abc = x_abc.copy()
    y_abc.append('$')
    plt.xticks(range(26), x_abc)
    plt.yticks(range(27), y_abc)
    plt.title('Bi-Grams heat map')
    plt.savefig('heat_map')
    plt.show()


class_numbers = 26
feature_length = 128
phi_length1 = class_numbers * feature_length
phi_length2 = phi_length1 + 27*26

# read_data(train=True)
# read_data(train=False)
# w1 = train_multiclass_perceptron(np.load('letters.npy'), np.load('images.npy'))
# print(test_multiclass_perceptron(w1, np.load('test_letters.npy'), np.load('test_images.npy')))

# w2 = train_structured1(np.load('letters.npy'), np.load('images.npy'))
# print(test_structured1(w2, np.load('test_letters.npy'), np.load('test_images.npy')))

# w3 = train_structured2(np.load('letters.npy'), np.load('images.npy'), np.load('word_ids.npy'))
# print(test_structured2(w3, np.load('test_letters.npy'), np.load('test_images.npy'), np.load('test_word_ids.npy')))
# np.save('W', w3)

# draw_heat_map(W3)

# num_to_letter = dict(zip(range(26), string.ascii_lowercase))
# data = np.loadtxt('multiclass.pred')
# new_data = [num_to_letter[i] for i in data]
#
# f = open("multiclass.pred", "w")
# for i in new_data[:-1]:
#     f.write(i + '\n')
# f.write(new_data[-1])
# f.close()
#
# data = np.loadtxt('structured.pred')
# new_data = [num_to_letter[i] for i in data]
#
# f = open("structured.pred", "w")
# for i in new_data[:-1]:
#     f.write(i + '\n')
# f.write(new_data[-1])
# f.close()