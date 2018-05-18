import numpy as np
import scipy.stats
import scipy.special
import glob
from pandas import DataFrame
import copy


""" The third lab will be devoted to the naive Bayesian classifier.
The dataset is already divided into 10 parts for cross-validation. The task is to classify spam.
Spam messages contain spmsg in their title, normal messages contain legit. 
The text of the letter itself consists of two parts: the subject and the body of the letter. 
All words are replaced by int corresponding to their index in some global dictionary (a kind of anonymization).
 Accordingly, you are required to build a naive Bayesian classifier and, in doing so,

1) Come up with, or test what you can do with the subject and body's letter to improve the quality of work.
2) How to take into account (or not to take into account) the words that may occur in the training sample, 
but may not meet in the test sample and vice versa.
3) How to impose additional restrictions on your classifier so that good letters almost never get into spam, 
but at the same time, perhaps the overall quality of the classification hasn't decreased too much.
4) Understand how the classifier is arranged inside and be able to answer any questions about the theory associated
with it.

For writing the classifier it is allowed to use numpy, scipy and pandas. Cross-validation can be done by any library."""

def readEmail(filename):
    target = 'spam' if 'spmsg' in filename else 'legit'
    with open(filename, 'r') as f:
        subject = [int(word) for word in f.readline().replace('Subject: ', '').replace('\n', '').split()]
        f.readline()
        body = [int(word) for word in f.readline().split()]

    return target, subject, body


def readFolder(path):
    emails = {'target': [], 'subject': [], 'body': []}
    filenames = glob.glob(path + '*.txt')
    for filename in filenames:
        target, subject, body = readEmail(filename)
        emails['target'].append(target)
        emails['subject'].append(subject)
        emails['body'].append(body)

    return DataFrame(emails)


def getDatasets(path):
    folders = ['part%d'%i for i in range(1,11)]
    dataset = {}
    for folder in folders:
        dataset[folder] = readFolder(path + folder + '/')

    return dataset


def getWordsFrequencies(words):
    frequencies = {}
    for word in words:
        if word in frequencies.keys():
            frequencies[word] += 1
        else:
            frequencies[word] = 1
    return frequencies


def getWrodsTableFrequencies(table):
    table_frequencies = {}
    # summary = {}

    for words in table:
        frequencies = getWordsFrequencies(words)
        for key, val in frequencies.items():
            if key in table_frequencies.keys():
                table_frequencies[key].append(val)
            else:
                table_frequencies[key] = [val]

    return table_frequencies


def mergeDictionaries(dic1, dic2):
    result = copy.deepcopy(dic1)
    for key, val in dic2.items():
        if key in result.keys():
            result[key].extend(val)
        else:
            result[key] = val
    return result


def prepareDataset(dataset, column='body'):
    parts = ['part%d' % i for i in range(1, 11)]

    frequencies = dict()

    frequencies['spam'] = getClassFrequencies(column, dataset, parts, 'spam')
    frequencies['legit'] = getClassFrequencies(column, dataset, parts, 'legit')

    return frequencies


def getClassFrequencies(column, dataset, parts, target):
    frequencies = {}

    for part in parts:
        data = dataset[part].loc[dataset[part]['target'] == target]
        frequencies[part] = getWrodsTableFrequencies(data[column])
        #res_frequencies = mergeDictionaries(res_frequencies, frequencies)

    return frequencies


def sumarize(frequencies):
    summarized = {}
    for key, val in frequencies.items():
        summarized[key] = (np.mean(val), np.std(val))
    return summarized


def getProbability(x, mu, sigma):
    if sigma == 0:
        return 1
    # prob = (scipy.special.erf((x-mu)/(sigma*(2**(1/2)))) + 1)/2
    prob = scipy.stats.norm(mu, sigma).cdf(x)
    # print(prob)
    return prob


def classMembershipProbab(test_freq, learning_data):
    # probability = 1
    probabs = []
    for word, freq in test_freq.items():
        mu, sigma = learning_data.get(word, (0, 1))
        # probability += np.log(getProbability(freq, mu, sigma))
        probabs.append(getProbability(freq, mu, sigma))
    # return probability
    return np.sum(np.log((np.array(probabs))))


def getProbabilities(email, learning_dataset):
    spam_class_freq = learning_dataset[0]
    legit_class_freq = learning_dataset[1]
    probabilities = {}

    email_freq = getWordsFrequencies(email)

    probabilities['spam'] = classMembershipProbab(email_freq, spam_class_freq)
    probabilities['legit'] = classMembershipProbab(email_freq, legit_class_freq)

    return probabilities


def predict(subject, body, learning_frequencies_subject, learning_frequencies_body):
    probab_subject = getProbabilities(subject, learning_frequencies_subject)
    probab_body = getProbabilities(body, learning_frequencies_body)
    probab = dict()

    for key in ['spam', 'legit']:
        probab[key] = probab_body[key] + probab_subject[key]
        # print(probab_body[key], probab_subject[key])
    # print(probab)
    probab['spam'] += 20
    return min(probab, key=probab.get)


def validateForGivenPart(dataset, testing_part, subject_frequencies, body_frequencies):
    spam_freq_subject = subject_frequencies['spam']
    legit_freq_subject = subject_frequencies['legit']
    spam_freq_body = body_frequencies['spam']
    legit_freq_body = body_frequencies['legit']
    # print(spam_freq_body.keys())

    test_emails_body = dataset[testing_part]['body']
    test_emails_subject = dataset[testing_part]['subject']

    parts = ['part%d' % i for i in range(1,11)]
    learning_frequencies_spam_subject = dict()
    learning_frequencies_legit_subject = dict()
    learning_frequencies_spam_body = dict()
    learning_frequencies_legit_body = dict()
    for part in parts:
        if part == testing_part:
            continue
        learning_frequencies_spam_subject = mergeDictionaries(learning_frequencies_spam_subject, spam_freq_subject[part])
        learning_frequencies_legit_subject = mergeDictionaries(learning_frequencies_legit_subject, legit_freq_subject[part])
        learning_frequencies_spam_body = mergeDictionaries(learning_frequencies_spam_body, spam_freq_body[part])
        learning_frequencies_legit_body = mergeDictionaries(learning_frequencies_legit_body, legit_freq_body[part])

    learning_frequencies_spam_subject = sumarize(learning_frequencies_spam_subject)
    learning_frequencies_legit_subject = sumarize(learning_frequencies_legit_subject)
    learning_frequencies_spam_body = sumarize(learning_frequencies_spam_body)
    learning_frequencies_legit_body = sumarize(learning_frequencies_legit_body)


    idx = 0
    legit_spam_idx = 0
    results = []
    for subject, body in zip(test_emails_subject, test_emails_body):
        res = predict(subject, body, (learning_frequencies_spam_subject, learning_frequencies_legit_subject),
                                    (learning_frequencies_spam_body, learning_frequencies_legit_body))
        correct = 1 if res == dataset[testing_part]['target'][idx] else 0
        if dataset[testing_part]['target'][idx] == 'legit' and res == 'spam':
            legit_spam_idx += 1

        results.append(correct)
        idx += 1

    print('Legit classified as spam: ', legit_spam_idx)
    return np.mean(results)


if __name__ == "__main__":

    dataset = getDatasets("Bayes/pu1/")

    body_frequencies = prepareDataset(dataset, column='body')
    subject_frequencies = prepareDataset(dataset, column='subject')

    # temp = []
    # for part in ['part%d' % i for i in range(1, 11)]:
    #     print('Testing for', part, ': ', end='')
    #     res = validateForGivenPart(dataset, part, subject_frequencies, body_frequencies)
    #     print(round(res, 2))
    #     temp.append(res)
    # print('Average: ', np.mean(temp))

    res = validateForGivenPart(dataset, 'part1', subject_frequencies, body_frequencies)
    print(res)
