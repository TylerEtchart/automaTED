import csv, numpy as np


categories = ('Beautiful','Confusing','Courageous','Funny','Informative',
			  'Ingenious','Inspiring','Longwinded','Unconvincing',
			  'Fascinating','Jaw-dropping','Persuasive','OK','Obnoxious')


def vectorize(ratings):

	counts = [0]*len(categories)

	for i in eval(ratings):
		counts[categories.index(i['name'])] = i['count']

	return counts


def load_talks(fn):

	with open(fn, 'rb') as f:

		reader = csv.DictReader(f)
		talks = {}

		for r in reader:
			talks[r['url'].strip()] = r['transcript']

	return talks


def load_data(fn):

	talks = load_talks('transcripts.csv')

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
			data['profile'].append(vectorize(r['ratings']))
			data['talks'].append(talks[url])

		data['profile'] = np.asarray(data['profile'])

	return data


data = load_data('ted_main.csv')
