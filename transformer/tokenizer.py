import re 


REP_WORD = ["n't", " not"]


class Tokenizer:


    def __init__(self, sequences, vocab_size):
        self.most_word = {}
        words = []
        for seq in sequences:
            seq = seq.replace(REP_WORD[0], REP_WORD[1])
            words += self._clean_text(seq).split(" ")
        for word in words:
            word = word.lower()
            if len(word) == 0:
                continue
            if word not in self.most_word:
                self.most_word[word] = 1
            else:
                self.most_word[word] += 1
        item_words = [i for i in self.most_word.items()]
        item_words.sort(key=lambda x: x[1], reverse=True)
        item_words = item_words[:vocab_size-3]
        self.most_word = {}
        for item in item_words:
            self.most_word[item[0]] = item[1]
        self.token = {
            "<sos>": 1,
            "<eos>": 2,
            "<pad>": 0,
            "<unk>": 3
        }
        for idx, item in enumerate(item_words, 4):
            self.token[item[0]] = idx 
        

    def _clean_text(self, text):
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
        return text


    def encode(self, text, max_length):
        text = text.replace(REP_WORD[0], REP_WORD[1])
        text = self._clean_text(text).split(" ")
        text = text[:max_length-2]
        text = [word.lower() for word in text if len(word)]
        token = [self.token["<sos>"]] + [
            self.token[word] if word in self.token else 3 for word in text 
        ] + [self.token["<eos>"]]
        if len(token) != max_length:
            token += [self.token["<pad>"]] * (max_length - len(token))
        return token 
    

    def decode(self, token):
        sequence = []
        k = list(self.token.keys())
        v = list(self.token.values())
        for idx in token:
            if idx == 1 or idx == 2:
                continue
            find_idx = v.index(idx)
            sequence.append(k[find_idx])
        return sequence



# if __name__ == "__main__":
#     tokenizer = Tokenizer(["I don't understand"], 100)
#     print(tokenizer.token)
#     decode = tokenizer.decode([1,4,3,5,3,5,2])
#     print(decode)