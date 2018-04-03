import nltk
import numpy as np
from posRNN import PosRNN

class TemplateManager:

    def __init__(self):
        self.template = []
        self.sentence = ""
        self.template_position = 0
        self.pos_rnn = PosRNN(restore=True)
        self.list_of_punctuation = ['``', "''", '!', '#', '$', ')', '(', ',', '.', ':', '?']
        self.formatted_punctuation = {
            '``': " ``",
            "''": "''",
            '!': "!",
            '#': "#", 
            '$': " $", 
            ')': ")", 
            '(': " (", 
            ',': ",", 
            '.': ".", 
            ':': ":", 
            '?': "?"
        }


    def generate_template(self, primer, length):
        self.template = self.pos_rnn.sample(length, primer)
        self.sentence = primer
        self.template_position = 0


    def get_tags(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        token_tags = nltk.pos_tag(tokens)
        words = [t[0] for t in token_tags]
        tags = [t[1] for t in token_tags]
        return tags, words


    def add_word(self, word):
        tags, words = self.get_tags(word)

        local_template_position = self.template_position
        local_sentence = self.sentence

        tag_index = 0
        while tag_index < len(tags):
            if self.template[local_template_position] in self.list_of_punctuation:
                punctuation = self.formatted_punctuation[self.template[local_template_position]]
                local_sentence += punctuation
                local_template_position += 1
            else:
                local_sentence += " " + words[tag_index]
                tag_index += 1
                local_template_position += 1

        self.template_position = local_template_position
        self.sentence = local_sentence
        
        return True


    def match_word(self, word):
        tags, words = self.get_tags(word)

        local_template_position = self.template_position
        local_sentence = self.sentence

        tag_index = 0
        while tag_index < len(tags):
            if self.template[local_template_position] in self.list_of_punctuation:
                punctuation = self.formatted_punctuation[self.template[local_template_position]]
                local_sentence += punctuation
                local_template_position += 1
            else:
                if tags[tag_index] != self.template[local_template_position]:
                    return False
                else:
                    local_sentence += " " + words[tag_index]
                    tag_index += 1
                    local_template_position += 1

        self.template_position = local_template_position
        self.sentence = local_sentence
        
        return True

    
    def format_sentence(self):
        return self.sentence

        # tokens = sentence.split()
        # cur_index = 0
        # for f in self.format:
        #     for key, index_list in f.items():
        #         if key == "len":
        #             continue
        #         for index in index_list:
        #             if cur_index + index < len(tokens):
        #                 tokens[cur_index + index] += key

        #     tokens[cur_index] = tokens[cur_index].capitalize()
        #     cur_index += f["len"]
        # return " ".join(tokens)



if __name__ == "__main__":
    tm = TemplateManager()
    sentence = "he jumped over the moon then jumped alllll the way over to the monkey child"
    tm.generate_template(length=len(sentence.split()))
    matches = tm.match(sentence)
    formatted = tm.format_sentence(sentence)
    print(formatted)