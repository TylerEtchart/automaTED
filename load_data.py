import csv, numpy as np


categories = ('Beautiful','Confusing','Courageous','Funny','Informative',
			  'Ingenious','Inspiring','Longwinded','Unconvincing',
			  'Fascinating','Jaw-dropping','Persuasive','OK','Obnoxious')


def vectorize(ratings):

	counts = [0]*len(categories)

	for i in eval(ratings):
		counts[categories.index(i['name'])] = i['count']

	return counts


def load_meta(fn):

	with open(fn, 'rb') as f:

		reader = csv.DictReader(f)
		# print(reader.fieldnames)
		meta = {'url':[],'title':[],'views':[],'comments':[],'profile':[]}

		for r in reader:
			meta['url'].append(r['url'])
			meta['title'].append(r['title'])
			meta['views'].append(int(r['views']))
			meta['comments'].append(int(r['comments']))
			meta['profile'].append(vectorize(r['ratings']))

		meta['profile'] = np.asarray(meta['profile'])

	return meta


def load_talks(fn):

	with open(fn, 'rb') as f:

		reader = csv.DictReader(f)
		# print(reader.fieldnames)
		talks = {}

		for r in reader:
			talks[r['url']] = r['transcript']

	return talks


meta = load_meta('ted_main.csv')
talks = load_talks('transcripts.csv')