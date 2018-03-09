import nltk
import numpy as np

class TemplateManager:

    def __init__(self):
        self.define_parts_of_speech()
        self.define_templates()
        self.define_formats()
        self.template = []
        self.format = []


    def define_parts_of_speech(self):
        # list of tags: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
        self.noun = set(["NN", "NNS", "NNP", "NNPS", "PRP", "WP"])
        self.possesive = set(["POS", "PRP$", "WP$"])
        self.verb = set(["VB", "VBD", "VBN", "VBP", "VBZ"])
        self.article = set(["DT"])
        self.preposition = set(["IN"])


    # TODO: write more templates
    def define_templates(self):
        self.templates = []
        self.templates.append([self.noun, self.verb, self.preposition,
                               self.article, self.noun])


    # should mirror define_templates
    def define_formats(self):
        self.formats = []
        self.formats.append({".": [4], "len": 5})


    def generate_template(self, length):
        self.template = []
        self.format = []
        while len(self.template) < length:
            index = np.random.choice(range(len(self.templates)))
            self.template.extend(self.templates[index])
            self.format.append(self.formats[index])


    def get_tags(self, sentence):
        tokens = sentence.split()
        token_tags = nltk.pos_tag(tokens)
        tags = [t[1] for t in token_tags]
        return tags


    def match_latest(self, sentence):
        tags = self.get_tags(sentence)
        index = len(tags) - 1
        return tags[index] in self.template[index]


    def match(self, sentence):
        tags = self.get_tags(sentence)
        for i, tag in enumerate(tags):
            if tag not in self.template[i]:
                return False
        return True

    
    def format_sentence(self, sentence):
        tokens = sentence.split()
        cur_index = 0
        for f in self.format:
            for key, index_list in f.items():
                if key == "len":
                    continue
                for index in index_list:
                    if cur_index + index < len(tokens):
                        tokens[cur_index + index] += key

            tokens[cur_index] = tokens[cur_index].capitalize()
            cur_index += f["len"]
        return " ".join(tokens)



if __name__ == "__main__":
    tm = TemplateManager()
    sentence = "he jumped over the moon then jumped alllll the way over to the monkey child"
    tm.generate_template(length=len(sentence.split()))
    matches = tm.match(sentence)
    formatted = tm.format_sentence(sentence)
    print(formatted)