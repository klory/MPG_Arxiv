from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import math
import json
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import lmdb
from io import BytesIO
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import utils
from gensim.models.keyedvectors import KeyedVectors
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random

import sys
sys.path.append('../../')
from common import ROOT
from AttnGAN.code.miscc.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    # class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    # keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    return [real_imgs, captions, sorted_cap_lens,
            None, None]
#def prepare_data(data):
#    imgs, captions, captions_lens, class_ids, keys = data
#    # sort data by the length in a decreasing order
#    sorted_cap_lens, sorted_cap_indices = \
#        torch.sort(captions_lens, 0, True)
#
#    real_imgs = []
#    for i in range(len(imgs)):
#        print(i)
#        print(sorted_cap_indices)
#        imgs[i] = imgs[i][sorted_cap_indices]
#        if cfg.CUDA:
#            real_imgs.append(Variable(imgs[i]).cuda())
#        else:
#            real_imgs.append(Variable(imgs[i]))
#
#    captions = captions[sorted_cap_indices].squeeze()
#    class_ids = class_ids[sorted_cap_indices].numpy()
#    # sent_indices = sent_indices[sorted_cap_indices]
#    keys = [keys[i] for i in sorted_cap_indices.numpy()]
#    # print('keys', type(keys), keys[-1])  # list
#    if cfg.CUDA:
#        captions = Variable(captions).cuda()
#        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
#    else:
#        captions = Variable(captions)
#        sorted_cap_lens = Variable(sorted_cap_lens)
#
#    return [real_imgs, captions, sorted_cap_lens,
#            class_ids, keys]

def get_ingredients_wordvec(recipe, w2i, permute_ingrs=False, max_len=20):
    '''
    get the ingredients wordvec for the recipe, the 
    number of items might be different for different 
    recipe
    '''
    ingredients = recipe
    if permute_ingrs:
        ingredients = np.random.permutation(ingredients).tolist()
    vec = np.zeros([max_len], dtype=np.int)
    num_words = min(max_len, len(ingredients))
        
    for i in range(num_words):
        word = ingredients[i]
        if word not in w2i:
            word = '<other>'
        vec[i] = w2i[word]
        
    return vec, num_words
def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
                # re_img = transforms.CenterCrop(imsize[i])(re_img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key


    def __len__(self):
        return len(self.filenames)


def choose_one_image(rcp, img_dir):
    part = rcp['partition']
    image_infos = rcp['images']
    if part == 'train':
        # We do only use the first five images per recipe during training
        imgIdx = np.random.choice(range(min(5, len(image_infos))))
    else:
        imgIdx = 0

    loader_path = [image_infos[imgIdx]['id'][i] for i in range(4)]
    loader_path = os.path.join(*loader_path)
    if 'plus' in img_dir:
        path = os.path.join(img_dir, loader_path, image_infos[imgIdx]['id'])
    else:
        path = os.path.join(img_dir, part, loader_path, image_infos[imgIdx]['id'])
    img = Image.open(path)
    img = img.convert('RGB')
    return img

def load_recipes(file_path, part=None):
    with open(file_path, 'r') as f:
        info = json.load(f)
    if part:
        info = [x for x in info if x['partition']==part]
    return info
class AttnFoodDataset(data.Dataset):
    def __init__(
        self, 
        recipe_file=f'{ROOT}/data/Recipe1M/original_withImage.json', 
        img_dir=f'{ROOT}/data/Recipe1M/images',
        part='train', 
        food_type='salad',
        word2vec_file=f"{ROOT}/data/Recipe1M/word2vec_recipes.bin",
        transform=None, 
        permute_ingrs=False,
        cfg=None):
        self.transform = transform
        self.cfg=cfg
        self.recipe_file = recipe_file
        self.img_dir = img_dir
        self.permute_ingrs = permute_ingrs
        self.recipes = load_recipes(recipe_file, part)
        if food_type:
            self.recipes = [x for x in self.recipes if food_type.lower() in x['title'].lower()]   
        wv = KeyedVectors.load(word2vec_file, mmap='r')
        w2i = {w: i+2 for i, w in enumerate(wv.index2word)}
        self.w2i = w2i
        self.w2i['<other>'] = 1
        print('vocab size =', len(self.w2i))
        self.ixtoword, self.wordtoix,self.n_words=build_dictionary2(self.recipes,self.w2i)
        print("#####################")
        print(len(self.ixtoword))
        print(len(self.self.wordtoix))
        base_size=cfg.TREE.BASE_SIZE
        self.imsize=[]
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def _prepare_one_recipe(self, index):
        rcp = self.recipes[index]

        # title, n_words_in_title = get_title_wordvec(rcp, self.wordtoix) # np.int [max_len]
        # ingredients, n_ingrs = get_ingredients_wordvec(rcp, self.wordtoix, self.permute_ingrs) # np.int [max_len]
        # instructions, n_insts, n_words_each_inst = get_instructions_wordvec(rcp,self.wordtoix) # np.int [max_len, max_len]

        pil_img = choose_one_image(rcp, self.img_dir) # PIL [3, 224, 224]
        img = self.transform(pil_img)
        return torch.from_numpy(np.asarray(img))

    def __getitem__(self, index):
        all_idx = range(len(self.recipes))
        wrong_idx = np.random.choice(all_idx)
        while wrong_idx == index:
            wrong_idx = np.random.choice(all_idx)
        img = self._prepare_one_recipe(index)
        unpairedimg = self._prepare_one_recipe(wrong_idx)
        ret=[]
        for i in range(self.cfg.TREE.BRANCH_NUM):
            if i < (self.cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(self.imsize[i])(img)
            else:
                re_img = img
            ret.append(self.norm(re_img))

        if len(self.recipes[index]['ingredients'])>18:
            caption=self.recipes[index]['ingredients'][:18]
        else:
            caption=self.recipes[index]['ingredients']
        rev = []
        for w in caption:
            if w in self.wordtoix:
                rev.append(self.wordtoix[w])
            else:
                print(w)
                print("Error, w not in wordtoix")

        sent_caption = np.asarray(rev).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros(self.cfg.TEXT.WORDS_NUM, dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM

        return ret,x,self.cfg.TEXT.WORDS_NUM, 0, 0

    def __len__(self):
        return len(self.recipes)


def build_dictionary2(recipes, w2i):
    rcp = recipes
    word_counts = defaultdict(float)
    for sent in rcp:
        for word in sent['ingredients']:
            if word not in w2i:
                word = '<other>'
            word_counts[word] += 1
    vocab = [w for w in word_counts if word_counts[w] >= 0]
    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    # print("wordtoixother")
    # print(wordtoix['<other>'])
    n_words=len(ixtoword)
    # print("n_words:  "+str(n_words))
    return ixtoword, wordtoix,len(ixtoword)





class Recipe1MDataset(data.Dataset):
    def __init__(
        self, 
        lmdb_file=f'{ROOT}/data/Recipe1M/Recipe1M.lmdb',
        recipe_file=f'{ROOT}/Recipe1M/recipes_withImage.json', 
        word2vec_file=f'{ROOT}/Recipe1M/word2vec_recipes.bin',
        part='', food_type='',
        transform=None, return_image=True,cfg=None):
        self.recipes = load_recipes(recipe_file, part)
        if food_type:
            self.recipes = [x for x in self.recipes if food_type.lower() in x['title'].lower()]
        assert part in ['', 'train', 'val', 'test'], "part has to be in ['', 'train', 'val', 'test']"
        assert food_type in ['', 'salad', 'cookie', 'muffin'], "part has to be in ['', 'salad', 'cookie', 'muffin']"

        dirname = os.path.dirname(lmdb_file)
        path = os.path.join(dirname, 'keys.json')
        with open(path, 'r') as f:
            self.keys = json.load(f)
        if part:
            self.keys = [x for x in self.keys if x['partition']==part]
        if food_type:
            self.keys = [x for x in self.keys if food_type.lower() in x['title'].lower()]
        self.cfg=cfg
        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        # with self.env.begin(write=False) as txn:
        #     self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        wv = KeyedVectors.load(word2vec_file, mmap='r')
        w2i = {w: i+2 for i, w in enumerate(wv.index2word)}
        self.w2i = w2i
        self.w2i['<other>'] = 1
        print('vocab size =', len(self.w2i))
        self.ixtoword, self.wordtoix,self.n_words=build_dictionary2(self.recipes,self.w2i)
        print("#####################")
        print(len(self.ixtoword))
        print(len(self.wordtoix))
        base_size=cfg.TREE.BASE_SIZE
        self.imsize=[]
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.resolution = self.cfg.TREE.BASE_SIZE

        assert transform!=None, 'transform can not be None!'
        self.transform = transform

        self.return_image = return_image

    def __len__(self):
        return len(self.keys)

    def _load_recipe(self, rcp):
        rcp_id = rcp['id']

        with self.env.begin(write=False) as txn:
            key = f'title-{rcp_id}'.encode('utf-8')
            title = txn.get(key).decode('utf-8')

            key = f'ingredients-{rcp_id}'.encode('utf-8')
            ingredients = txn.get(key).decode('utf-8')

            key = f'instructions-{rcp_id}'.encode('utf-8')
            instructions = txn.get(key).decode('utf-8')

            # key = f'{self.resolution}-{rcp_id}'.encode('utf-8')
            key = f'256-{rcp_id}'.encode('utf-8')
            img_bytes = txn.get(key)

        txt = list(ingredients)
        if len(txt)>18:
            caption=txt[:18]
        else:
            caption=txt
        rev = []
        for w in caption:
            if w in self.wordtoix:
                rev.append(self.wordtoix[w])
            else:
                print(w)
                print("Error, w not in wordtoix")
        sent_caption = np.asarray(rev).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)   
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        ret=[]
        for i in range(self.cfg.TREE.BRANCH_NUM):
            if i < (self.cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(self.imsize[i])(img)
            else:
                re_img = img
            ret.append(self.norm(re_img))


        return x, ret

    def __getitem__(self, index):
        rcp_key = self.keys[index]
        txt, imgs = self._load_recipe(rcp_key)

        if not self.return_image:
            return txt
        
        all_idx = range(len(self.keys))
        wrong_idx = np.random.choice(all_idx)
        while wrong_idx == index:
            wrong_idx = np.random.choice(all_idx)
        _, wrong_img = self._load_recipe(self.keys[wrong_idx])

        return imgs,txt,self.cfg.TEXT.WORDS_NUM, 0, 0



class Pizza10Dataset(data.Dataset):
    def __init__(
        self, 
        data_dir=f'{ROOT}/pizzaGANdata_old/', 
        transform=None,part="train",cfg=None,base_size=64):
        with open(f"{ROOT}/data/pizzaGANdata_new_concise/categories.txt", 'r') as f:
            self.categories = np.asarray(f.read().strip().split('\n'))
        with open(f"{ROOT}/data/pizzaGANdata_new_concise/imageLabels.txt", 'r') as f:
            self.labels = f.read().strip().split('\n')
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.transform = transform
        self.cfg=cfg
        self.img_dir = os.path.join(data_dir, 'images/')
        self.ixtoword, self.wordtoix,self.n_words=self.build_dictionary()
        self.split = part
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __len__(self):
        return len(self.labels)
    def build_dictionary(self):

        vocab = self.categories.tolist()
        vocab.append("<other>")
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        # print("#"*100)
        # print(len(ixtoword))
        return ixtoword,wordtoix, len(ixtoword)
    def _load_image_from_index(self, index):
        
        img_path = os.path.join(self.img_dir, f'{index+1:>05d}.jpg')
        return get_imgs(img_path, self.imsize, bbox=None,transform=self.transform, normalize=self.norm),index
    def get_caption(self, sent_caption):

        sent_caption=sent_caption[sent_caption != 0]


        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    def __getitem__(self, index):
        paired_img,img_index = self._load_image_from_index(index)
        label = [int(x) for x in self.labels[img_index].split()]
        ingredients = [self.categories[i] for i in np.asarray(label).nonzero()[0]]
        if not ingredients:
            ingredients = ["<other>"]

        ingredients,_=get_ingredients_wordvec(ingredients,self.wordtoix)
        sent_caption = np.asarray(ingredients).astype('int64')
        x, x_len = self.get_caption(sent_caption)
        # zero_count = np.sum(sent_caption == 1 )
        # num_words = len(sent_caption)   
        # x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        # # if num_words <= cfg.TEXT.WORDS_NUM:
        # #     x[:num_words, 0] = sent_caption
        # x[:, 0] = sent_caption[:self.cfg.TEXT.WORDS_NUM]
        # print(x_len)
        return paired_img,x,x_len, 0, 0



class Pizza10DatasetByRecipe(data.Dataset):
    def __init__(
        self, 
        data_dir=f'{ROOT}/data/pizzaGANdata_new_concise/', 
        transform=None,part="train",cfg=None,base_size=64):
        with open(data_dir+"categories.txt", 'r') as f:
            self.categories = np.asarray(f.read().strip().split('\n'))
        with open(data_dir+"imageLabels.txt", 'r') as f:
            self.labels = f.read().strip().split('\n')
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.transform = transform
        self.cfg=cfg
        self.img_dir = os.path.join('{ROOT}/data/pizzaGANdata_old/', 'images/')
        self.ixtoword, self.wordtoix,self.n_words=self.build_dictionary()
        self.split = part
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        with open(data_dir+"imageLabels.txt", 'r') as f:
            list1 = f.readlines()
            label_dict= {}
            for idx,line in enumerate(list1):
                line = line.strip()
                line = line.rstrip('\n')
                if not line in label_dict:
                    label_dict[line] = []
                label_dict[line].append(idx)
            for k in list(label_dict.keys()):
                if len(label_dict[k])<5:
                    label_dict.pop(k)
            self.label_list = list(label_dict.values())
        # self.label_list = get_pizza_recipe_list("{ROOT}/data/pizzaGANdata_new_concise/imageLabels.txt")
        print(self.__len__())
    def __len__(self):
        return len(self.label_list)
    def build_dictionary(self):

        vocab = self.categories.tolist()
        vocab.append("<other>")
        # vocab = ["pepperoni","bacon","mushrooms","onion","peppers","blackolives","tomatoes","basil","arugula","corn","<other>"]
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        # print("#"*100)
        # print(len(ixtoword))
        return ixtoword,wordtoix, len(ixtoword)
    def _load_image_from_index(self, index):
        image_infos = self.label_list[index]
        # if self.split=="train":
        #     index = image_infos[int(2*np.random.random())]
        # elif self.split=="val":
        #     index = image_infos[3+int(2*np.random.random())]
        if self.split=="train":
            index = image_infos[int(3*np.random.random()*len(image_infos)/5)]
        elif self.split=="val":
            index = image_infos[math.floor(3*len(image_infos)/5)+int(2*np.random.random()*len(image_infos)/5)]

        img_path = os.path.join(self.img_dir, f'{index+1:>05d}.jpg')
        return get_imgs(img_path, self.imsize, bbox=None,transform=self.transform, normalize=self.norm),index
    def get_caption(self, sent_caption):

        sent_caption=sent_caption[sent_caption != 0]


        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    def __getitem__(self, index):
        paired_img,img_index = self._load_image_from_index(index)
        label = [int(x) for x in self.labels[img_index].split()]
        ingredients = [self.categories[i] for i in np.asarray(label).nonzero()[0]]
        if not ingredients:
            ingredients = ["<other>"]

        ingredients,_=get_ingredients_wordvec(ingredients,self.wordtoix)
        sent_caption = np.asarray(ingredients).astype('int64')
        x, x_len = self.get_caption(sent_caption)
        return paired_img,x,x_len, 0, 0