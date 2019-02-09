import re
import math
from collections import Counter

filename_en_1 = 'TrainingCorpusENandFR/en-moby-dick.txt'
filename_en_2 = 'TrainingCorpusENandFR/en-the-little-prince.txt'
filename_fr_1 = 'TrainingCorpusENandFR/fr-le-petit-prince.txt'
filename_fr_2 = 'TrainingCorpusENandFR/fr-vingt-mille-lieues-sous-les-mers.txt'
filename_gr_1 = 'TrainingCorpusENandFR/gr_Adalbert_Stifter_Der_Nachsommer.txt'
filename_un = 'TrainingCorpusENandFR/sentence'

cal_s_time = 0
cal_e_time = 0
load_s_time = 0
load_e_time = 0


def dataLoadingProcessing(filename):
    with open(filename, encoding="utf8", errors='ignore') as file:
        contents = file.read().replace('\n', '')
        file.close()
        sent = [re.sub(r"[^a-zA-Z]+", '', k) for k in contents.split("\n")]
        return sent


def ngrams(text, n=1):
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def create_ngram_features(sent, n=3):
    chr = [c for c in sent]
    ngram_vocab = ngrams(chr, n)
    if n == 1:
        ngramList = [ng[0].lower() for ng in ngram_vocab]
    elif n == 2:
        ngramList = [(x.lower(), y.lower()) for x, y in ngram_vocab]
    elif n == 3:
        ngramList = [(x.lower(), y.lower(), z.lower()) for x, y, z in ngram_vocab]
    elif n == 4:
        ngramList = [(a.lower(), b.lower(), c.lower(), d.lower()) for a, b, c, d in ngram_vocab]
    elif n == 5:
        ngramList = [(a.lower(), b.lower(), c.lower(), d.lower(), e.lower()) for a, b, c, d, e in ngram_vocab]
    return ngramList


#
sent_en1 = dataLoadingProcessing(filename_en_1)
sent_en2 = dataLoadingProcessing(filename_en_2)
sent_fr1 = dataLoadingProcessing(filename_fr_1)
sent_fr2 = dataLoadingProcessing(filename_fr_2)
sent_gr1 = dataLoadingProcessing(filename_gr_1)

sent_en = [sent_en1[0] + sent_en2[0]]
sent_fr = [sent_fr1[0] + sent_fr2[0]]
sent_gr = [sent_gr1[0]]


def calculate_prob(sent, delta=.5):
    en_data_uni = []
    en_data_bi = []
    en_data_tri = []
    en_data_qua = []
    en_data_qui = []

    for f in sent:
        en_data_uni.append((create_ngram_features(f, 1)))
        en_data_bi.append((create_ngram_features(f, 2)))
        en_data_tri.append((create_ngram_features(f, 3)))
        en_data_qua.append((create_ngram_features(f, 4)))
        en_data_qui.append((create_ngram_features(f, 5)))

    uni = en_data_uni[0]
    total_len_uni = len(uni)
    bi = en_data_bi[0]
    tri = en_data_tri[0]
    qua = en_data_qua[0]
    qui = en_data_qui[0]

    char_freq_uni = {}
    cnt = Counter(uni)
    for k, v in cnt.items():
        char_freq_uni[k] = v

    char_freq_bi = {}
    cnt = Counter(bi)
    for k, v in cnt.items():
        char_freq_bi[k] = v

    char_freq_tri = {}
    cnt = Counter(tri)
    for k, v in cnt.items():
        char_freq_tri[k] = v

    char_freq_qua = {}
    cnt = Counter(qua)
    for k, v in cnt.items():
        char_freq_qua[k] = v

    char_freq_qui = {}
    cnt = Counter(qui)
    for k, v in cnt.items():
        char_freq_qui[k] = v

    char_freq_uni = dict(sorted(char_freq_uni.items()))

    letter_prob_uni = {ch[0]: (char_freq_uni[ch[0]] + .5) / (total_len_uni + .5 * 26) for ch in char_freq_uni}
    letter_prob_uni['default'] = 1 / 26
    letter_prob_bi = {
        (ch[0], ch[1]): ((char_freq_bi[(ch[0], ch[1])] + delta) / (char_freq_uni[ch[0]] + delta * 26 * 26))
        for ch in char_freq_bi}
    letter_prob_bi['default'] = 1 / (26 * 26)
    letter_prob_tri = {(ch[0], ch[1], ch[2]): (
            (char_freq_tri[(ch[0], ch[1], ch[2])] + delta) / (char_freq_bi[(ch[0], ch[1])] + delta * 26 * 26 * 26))
        for ch in char_freq_tri}
    letter_prob_tri['default'] = 1 / (26 * 26 * 26)

    letter_prob_qua = {(ch[0], ch[1], ch[2], ch[3]): (
            (char_freq_qua[(ch[0], ch[1], ch[2], ch[3])] + delta) / (
            char_freq_tri[(ch[0], ch[1], ch[2])] + delta * 26 * 26 * 26 * 26))
        for ch in char_freq_qua}
    letter_prob_qua['default'] = 1 / (26 * 26 * 26 * 26)

    letter_prob_qui = {(ch[0], ch[1], ch[2], ch[3], ch[4]): (
            (char_freq_qui[(ch[0], ch[1], ch[2], ch[3], ch[4])] + delta) / (
            char_freq_qua[(ch[0], ch[1], ch[2], ch[3])] + delta * 26 * 26 * 26 * 26))
        for ch in char_freq_qui}
    letter_prob_qui['default'] = 1 / (26 * 26 * 26 * 26 * 26)

    return letter_prob_uni, letter_prob_bi, letter_prob_tri, letter_prob_qua, letter_prob_qui


prob_uni_en, prob_bi_en, prob_tri_en, prob_qua_en, prob_qui_en = calculate_prob(sent_en, .5)
prob_uni_fr, prob_bi_fr, prob_tri_fr, prob_qua_fr, prob_qui_fr = calculate_prob(sent_fr, .5)
prob_uni_gr, prob_bi_gr, prob_tri_gr, prob_qua_gr, prob_qui_gr = calculate_prob(sent_gr, .5)

data_uni = []
data_bi = []
data_tri = []


def display_prob_uni(x):
    if x in prob_uni_fr:
        un_pr_fr = prob_uni_fr[x]
    else:
        un_pr_fr = prob_uni_fr['default']
    if x in prob_uni_en:
        un_pr_en = prob_uni_en[x]
    else:
        un_pr_en = prob_uni_en['default']
    if x in prob_uni_gr:
        un_pr_gr = prob_uni_gr[x]
    else:
        un_pr_gr = prob_uni_gr['default']

    return un_pr_fr, un_pr_en, un_pr_gr


def display_prob_bi(x):
    if x in prob_bi_fr:
        bi_pr_fr = prob_bi_fr[x]
    else:
        bi_pr_fr = prob_bi_fr['default']
    if x in prob_bi_en:
        bi_pr_en = prob_bi_en[x]
    else:
        bi_pr_en = prob_bi_en['default']
    if x in prob_bi_gr:
        bi_pr_gr = prob_bi_gr[x]
    else:
        bi_pr_gr = prob_bi_gr['default']

    return bi_pr_fr, bi_pr_en, bi_pr_gr


def display_prob_tri(x):
    if x in prob_tri_fr:
        tri_pr_fr = prob_tri_fr[x]
    else:
        tri_pr_fr = prob_tri_fr['default']
    if x in prob_tri_en:
        tri_pr_en = prob_tri_en[x]
    else:
        tri_pr_en = prob_tri_en['default']
    if x in prob_tri_gr:
        tri_pr_gr = prob_tri_gr[x]
    else:
        tri_pr_gr = prob_tri_gr['default']

    return tri_pr_fr, tri_pr_en, tri_pr_gr


def display_prob_qua(x):
    if x in prob_qua_fr:
        qua_pr_fr = prob_qua_fr[x]
    else:
        qua_pr_fr = prob_qua_fr['default']
    if x in prob_qua_en:
        qua_pr_en = prob_qua_en[x]
    else:
        qua_pr_en = prob_qua_en['default']
    if x in prob_qua_gr:
        qua_pr_gr = prob_qua_gr[x]
    else:
        qua_pr_gr = prob_qua_gr['default']

    return qua_pr_fr, qua_pr_en, qua_pr_gr


def display_prob_qui(x):
    if x in prob_qui_fr:
        qui_pr_fr = prob_qui_fr[x]
    else:
        qui_pr_fr = prob_qui_fr['default']
    if x in prob_qui_en:
        qui_pr_en = prob_qui_en[x]
    else:
        qui_pr_en = prob_qui_en['default']
    if x in prob_qui_gr:
        qui_pr_gr = prob_qui_gr[x]
    else:
        qui_pr_gr = prob_qui_gr['default']

    return qui_pr_fr, qui_pr_en, qui_pr_gr


def console_trace(sent_un, debug=False):
    data_list_tri = [(create_ngram_features(sent_un, 3))]
    value_en = 0
    value_fr = 0
    value_gr = 0

    for x in data_list_tri[0]:
        tri_pr_fr, tri_pr_en, tri_pr_gr = display_prob_tri(x)
        value_fr += math.log10(tri_pr_fr)
        value_en += math.log10(tri_pr_en)
        value_gr += math.log10(tri_pr_gr)

    lang = ""
    if (value_en > value_fr) and (value_en > value_gr):
        lang = "English"
    elif (value_fr > value_en) and (value_fr > value_gr):
        lang = "French"
    elif (value_gr > value_en) and (value_gr > value_fr):
        lang = "German"
    print('According to the Trigram model, the sentence is in ' + lang)
    data_list_qua = []
    data_list_qua.append((create_ngram_features(sent_un, 4)))
    value_en = 0
    value_fr = 0
    value_gr = 0

    for x in data_list_qua[0]:
        qua_pr_fr, qua_pr_en, qua_pr_gr = display_prob_qua(x)
        value_fr += math.log10(qua_pr_fr)
        value_en += math.log10(qua_pr_en)
        value_gr += math.log10(qua_pr_gr)
    if debug:
        print('For English: ' + str(value_en))
        print('For French: ' + str(value_fr))
        print('For German: ' + str(value_gr))
    lang = ""
    if (value_en > value_fr) and (value_en > value_gr):
        lang = "English"
    elif (value_fr > value_en) and (value_fr > value_gr):
        lang = "French"
    elif (value_gr > value_en) and (value_gr > value_fr):
        lang = "German"
    print('According to the quart model, the sentence is in ' + lang)

    data_list_qui = [(create_ngram_features(sent_un, 5))]
    value_en = 0
    value_fr = 0
    value_gr = 0

    for x in data_list_qui[0]:
        qui_pr_fr, qui_pr_en, qui_pr_gr = display_prob_qui(x)
        value_fr += math.log10(qui_pr_fr)
        value_en += math.log10(qui_pr_en)
        value_gr += math.log10(qua_pr_gr)
    if debug:
        print('For English: ' + str(value_en))
        print('For French: ' + str(value_fr))
        print('For German: ' + str(value_gr))
    lang = ""
    if (value_en > value_fr) and (value_en > value_gr):
        lang = "English"
    elif (value_fr > value_en) and (value_fr > value_gr):
        lang = "French"
    elif (value_gr > value_en) and (value_gr > value_fr):
        lang = "German"
    print('According to the quint model, the sentence is in ' + lang)


def dump_trace(line, output, debug=False):
    sent = [re.sub(r"[^a-zA-Z]+", '', k) for k in line.split("\n")]
    sent_un = sent[0]
    data_list_uni = [(create_ngram_features(sent_un, 1))]
    value_en = 0
    value_fr = 0
    value_gr = 0

    print('\n' + str(output) + ') ' + line.rstrip())
    file = open('data' + str(output) + '.txt', 'w')
    file.write(str(line))
    file.write('\nUNIGRAM MODEL:')
    for x in data_list_uni[0]:
        un_pr_fr, un_pr_en, un_pr_gr = display_prob_uni(x)
        value_fr += math.log10(un_pr_fr)
        value_en += math.log10(un_pr_en)
        value_gr += math.log10(un_pr_gr)
        file.write('\n\nUNIGRAM: ' + x)
        file.write(
            '\nFRENCH: P(' + x + ') = ' + str(un_pr_fr) + '==> log prob of sentence so far: ' + str(value_fr))
        file.write(
            '\nENGLISH: P(' + x + ') = ' + str(un_pr_en) + '==> log prob of sentence so far: ' + str(value_en))
        file.write(
            '\nOTHER: P(' + x + ') = ' + str(un_pr_gr) + '==> log prob of sentence so far: ' + str(value_gr))
    if debug == True:
        print('For English: ' + str(value_en))
        print('For French: ' + str(value_fr))
        print('For German: ' + str(value_gr))

    lang = ""
    if (value_en > value_fr) and (value_en > value_gr):
        lang = "English"
    elif (value_fr > value_en) and (value_fr > value_gr):
        lang = "French"
    elif (value_gr > value_en) and (value_gr > value_fr):
        lang = "German"

    print('According to the unigram model, the sentence is in ' + lang)
    file.write('\n\nAccording to the unigram model, the sentence is in ' + lang)
    data_list_bi = []
    data_list_bi.append((create_ngram_features(sent_un, 2)))
    value_en = 0
    value_fr = 0
    value_gr = 0
    file.write('\n- - - - - - - - - - - - - - - - - ')
    file.write('\nBIGRAM MODEL:')
    for x in data_list_bi[0]:
        bi_pr_fr, bi_pr_en, bi_pr_gr = display_prob_bi(x)
        value_fr += math.log10(bi_pr_fr)
        value_en += math.log10(bi_pr_en)
        value_gr += math.log10(bi_pr_gr)
        file.write('\n\nBIGRAM: ' + x[0] + x[1])
        file.write(
            '\nFRENCH: P(' + x[0] + '|' + x[1] + ') = ' + str(bi_pr_fr) + '==> log prob of sentence so far: ' + str(
                value_fr))
        file.write(
            '\nENGLISH: P(' + x[0] + '|' + x[1] + ') = ' + str(bi_pr_en) + '==> log prob of sentence so far: ' + str(
                value_en))
        file.write(
            '\nOTHER: P(' + x[0] + '|' + x[1] + ') = ' + str(bi_pr_gr) + '==> log prob of sentence so far: ' + str(
                value_gr))
    if debug == True:
        print('For English: ' + str(value_en))
        print('For French: ' + str(value_fr))
        print('For German: ' + str(value_gr))
    lang = ""
    if (value_en > value_fr) and (value_en > value_gr):
        lang = "English"
    elif (value_fr > value_en) and (value_fr > value_gr):
        lang = "French"
    elif (value_gr > value_en) and (value_gr > value_fr):
        lang = "German"
    print('According to the bigram model, the sentence is in ' + lang)
    file.write('\n\nAccording to the bigram model, the sentence is in ' + lang)
    file.close()
    console_trace(sent_un)


with open(filename_un) as file:
    output = 1
    for line in file:
        dump_trace(line, output)
        output += 1
