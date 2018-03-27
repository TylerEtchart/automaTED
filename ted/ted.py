import csv
import re
import collections
import numpy as np

class TED:

    def __init__(self, folder):
        self.categories = ('Beautiful','Confusing','Courageous','Funny','Informative',
                    'Ingenious','Inspiring','Longwinded','Unconvincing',
                    'Fascinating','Jaw-dropping','Persuasive','OK','Obnoxious')
        self.data = self.load_data(folder)


    def vectorize(self, ratings):
        counts = [0]*len(self.categories)
        for i in eval(ratings):
            counts[self.categories.index(i['name'])] = i['count']
        return counts


    def load_talks(self, fn):
        with open(fn, 'rb') as f:
            reader = csv.DictReader(f)
            talks = {}
            for r in reader:
                talks[r['url'].strip()] = r['transcript']
        return talks


    def load_data(self, folder):
        data_filename = folder + 'ted_main.csv'
        talks_filename = folder + 'transcripts.csv'
        talks = self.load_talks(talks_filename)

        with open(data_filename, 'rb') as f:
            reader = csv.DictReader(f)
            data = {'url':[],'title':[],'views':[],'comments':[],'profile':[], 'talks':[]}

            for r in reader:
                url = r['url'].strip()
                
                if url not in talks:
                    continue

                data['url'].append(url)
                data['title'].append(r['title'])
                data['views'].append(int(r['views']))
                data['comments'].append(int(r['comments']))
                data['profile'].append(self.vectorize(r['ratings']))
                data['talks'].append(talks[url])

            data['profile'] = np.asarray(data['profile'])

        return data


    def normalize_profiles_across_everything(self):
        profiles = np.array(self.data['profile']).astype(float)
        profile_sums = profiles.sum(axis=0)
        self.profiles = (profiles.T / profile_sums[:, np.newaxis]).T


    def normalize_profiles_locally(self):
        profiles = np.array(self.data['profile']).astype(float)
        profile_sums = profiles.sum(axis=1)
        self.profiles = (profiles / profile_sums[:, np.newaxis])


    def normalize_views(self):
        views = np.array(self.data['views']).astype(float)
        self.views = views / np.max(views)


    def generate_vocab(self):
        # prepare data
        text = self.data['talks']
        keep_pattern = re.compile("[^0-9a-zA-Z'\-\s]")
        strip_pattern = re.compile("\([a-zA-Z]*\)")
        frequency_constraint = 60

        # frequency_constraint --> vocab_size
        # 0 --> 71381
        # 60 --> 5131
        # 100 --> 3534
        # 200 --> 2026

        # strip talks
        self.stripped_talks = []
        self.words = []
        for i in range(len(text)):
            talk = text[i]
            talk = strip_pattern.sub(" ", talk)
            talk = keep_pattern.sub(" ", talk)
            talk = re.split('\s*', talk)
            talk = [t.lower() for t in talk if t != ""]
            self.stripped_talks.append(talk)
            self.words.extend(talk)

        # generate vocab
        vocab_counter = collections.Counter(self.words)
        self.vocab_list = [word for word, frequency in vocab_counter.items() if frequency >= frequency_constraint]
        self.vocab_size = len(self.vocab_list)

        # final strip of talks
        vocab_set = set(self.vocab_list)
        for i in range(len(self.stripped_talks)):
            self.stripped_talks[i] = [word for word in self.stripped_talks[i] if word in vocab_set]
        self.words = [word for word in self.words if word in vocab_set]
        self.talk_lengths = [len(talk) for talk in self.stripped_talks]

        # get the counts of the talks
        self.talk_counts = []
        for i in range(len(self.talk_lengths)):
            if i == 0:
                self.talk_counts.append(self.talk_lengths[i])
            else:
                self.talk_counts.append(self.talk_lengths[i] + self.talk_counts[i - 1])
        

    def to_txt(self):
        text = " ".join(self.data['talks'])
        with open("input.txt", "w") as file:
            file.write(text)


if __name__ == "__main__":
    t = TED("")
    t.generate_vocab()
    print(t.vocab_size)
