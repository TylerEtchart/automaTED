import csv
import re
import collections
import numpy as np

class TED:

    def __init__(self):
        self.categories = ('Beautiful','Confusing','Courageous','Funny','Informative',
                    'Ingenious','Inspiring','Longwinded','Unconvincing',
                    'Fascinating','Jaw-dropping','Persuasive','OK','Obnoxious')
        self.data = self.load_data('ted_main.csv')
        self.vocab = self.generate_vocab()


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


    def load_data(self, fn):

        talks = self.load_talks('transcripts.csv')

        with open(fn, 'rb') as f:

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


    def generate_vocab(self):
        # prepare data
        text = self.data['talks']
        keep_pattern = re.compile("[^0-9a-zA-Z'\-\s]")
        strip_pattern = re.compile("\([a-zA-Z]*\)")
        frequency_constraint = 60

        # strip talks
        self.stripped_talks = []
        for i in range(len(text)):
            talk = text[i]
            talk = strip_pattern.sub(" ", talk)
            talk = keep_pattern.sub(" ", talk)
            talk = re.split('\s*', talk)
            talk = [t.lower() for t in talk if t != ""]
            self.stripped_talks.extend(talk)

        # generate vocab
        vocab_counter = collections.Counter(self.stripped_talks)
        self.vocab_list = [word for word, frequency in vocab_counter.items() if frequency >= frequency_constraint]
        self.vocab_size = len(self.vocab_list)

        # final strip of talks
        vocab_set = set(self.vocab_list)
        self.stripped_talks = [word for word in self.stripped_talks if word in vocab_set]

    def to_txt(self):
        text = " ".join(self.data['talks'])
        with open("input.txt", "w") as file:
            file.write(text)





        
t = TED()
t.to_txt()