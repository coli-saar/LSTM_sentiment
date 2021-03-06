import torch
import json
import sys
import numpy as np
import filereader
import re
from torch.autograd import Variable
from torch.utils.data import Dataset

import torchwordemb


class YelpReviewsOneHotChars(Dataset):
    def __init__(self, path):
        self.reader = filereader.FileReader(path)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        line = self.reader[item]
        data = json.loads(line)
        features = data["text"]
        features = [ord(c) for c in features]
        line_array = np.zeros([len(features), 256], dtype="float32")
        for i, j in enumerate(features):
            line_array[i, j] = 1
        keys = ["stars", "useful", "cool", "funny"]
        targets = np.array([float(data[i]) for i in keys], dtype='float32')

        return Variable(torch.from_numpy(line_array)), Variable(torch.from_numpy(targets))


class YelpReviewsCharIdxes(Dataset):
    """ Dataset built for loading yelp reviews into an indexed format.
    Should be used together with models that learn their own embeddings.
    """
    def __init__(self, path):
        self.reader = filereader.FileReader(path)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        line = self.reader[item]
        data = json.loads(line)
        features = data["text"]
        features = np.array([ord(c) for c in features], dtype="int64")
        keys = ["stars", "useful", "cool", "funny"]
        targets = np.array([float(data[i]) for i in keys], dtype='float32')

        return Variable(torch.from_numpy(features)), Variable(torch.from_numpy(targets))


class YelpReviewsWordHash(Dataset):
    """ Dataset built for loading yelp reviews into an indexed format.
    Should be used together with models that learn their own embeddings.
    """
    def __init__(self, path):
        self.reader = filereader.FileReader(path)
        self.pattern = re.compile('[^ \w]+')

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        line = self.reader[item]
        data = json.loads(line)
        features = data["text"]
        features = self.pattern.sub('', features.lower())
        features = np.array([hash(i) % 256 for i in features.split(" ")], dtype="int64")
        keys = ["stars", "useful", "cool", "funny"]
        targets = np.array([float(data[i]) for i in keys], dtype='float32')

        return Variable(torch.from_numpy(features)), Variable(torch.from_numpy(targets))


class UserDict:
    def __init__(self):
        self.user_to_id = dict()
        self.id_to_user = []

    def lookup(self, user):
        if user in self.user_to_id:
            return self.user_to_id[user]
        else:
            id = len(self.id_to_user)
            self.user_to_id[user] = id
            self.id_to_user.append(user)
            return id

    def num_users(self):
        return len(self.id_to_user)

class GlovePretrained50d(Dataset):
    """ Dataset that converts the sentences using pretrained 50-dimensional glove word vectors.
    """

    def __init__(self, path, glove_path="./glove.6B.50d.txt"):
        self.reader = filereader.FileReader(path)
        self.pattern = re.compile('[^ \w]+')
        self.userdict = UserDict()

        print("Reading word vectors...")
        self.vocab, self.vec = torchwordemb.load_glove_text(glove_path)

        print("Collecting user IDs ...")
        for i in range(len(self.reader)):
            line = self.reader[i]
            data = json.loads(line)
            user = data["user_id"]
            userid = self.userdict.lookup(user)

        print("Done!")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        line = self.reader[item]
        data = json.loads(line)

        user = data["user_id"]
        userid = self.userdict.lookup(user)

        features = data["text"]
        features = self.pattern.sub('', features.lower())

        remapped = []
        for word in features.split(" "):
            if word in self.vocab:
                remapped.append(self.vec[self.vocab[word]])
            else:
                remapped.append(torch.zeros(50))
        features = torch.stack(remapped)

        keys = ["stars", "useful", "cool", "funny"]
        targets = np.array([float(data[i]) for i in keys], dtype='float32')

        return Variable(features), Variable(torch.from_numpy(targets)), userid


class RandomData(Dataset):
    """ Random data generator, designed for speed checks """

    def __init__(self, path, output_len):
        self.len = 0
        self.output_len = output_len
        print("Dataset: going through all reviews...")
        with open(path) as f:
            for line in f:
                self.len += 1
        print("Dataset: Is ready!")

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return Variable(torch.randn(self.output_len, 256)), Variable(torch.ones(4))


if __name__ == "__main__":
    dataset = YelpReviewsOneHotChars(sys.argv[1])
    print(len(dataset))
    print(dataset[1])
    import code
    code.interact(local=locals())
                

