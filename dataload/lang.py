import torch
from etc import defaultsetting as ds

defaultTokken = {
    ds.TOKKEN_PAD_IDX: ds.TOKKEN_PAD,
    ds.TOKKEN_UNK_IDX: ds.TOKKEN_UNK
}

class Lang:
    def __init__(self, name, defaultTokken=defaultTokken):
        self.name = name
        self.tag_PADid = ds.TOKKEN_PAD_IDX
        
        self.word2index = {}
        self.word2count = {}
        self.index2word = defaultTokken.copy() # {tag_UNK: "<UNK>", tag_PAD: "<PAD>"}
        self.n_words = len(self.index2word) # 2  # UNK 와 PAD 포함
        self.max_len_word = 0
        
        self.char2index = {}
        self.char2count = {}
        self.index2char = defaultTokken.copy() # {tag_UNK: "<UNK>", tag_PAD: "<PAD>"}
        self.n_chars = len(self.index2char) # 2  # UNK 와 PAD 포함
        self.max_len_char = 0
        
    def addSentence(self, sentence):
        l = len(sentence.split(' '))
        if l > self.max_len_word:
            self.max_len_word = l
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
        l = len(word)
        if l > self.max_len_char:
            self.max_len_char = l
        for char in word:
            self.addChar(char)
            
    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1
            
    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]

    def sentenceFromIndexes(self, indexes):
        return [self.index2word[idx] for idx in indexes]

    def tensorFromSentence(self, sentence, device='cpu'):
        indexes = self.indexesFromSentence(sentence)
        PAD_indexes = [self.tag_PADid for i in range(self.max_len_word - len(indexes))]
        indexes.extend(PAD_indexes)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(self.max_len_word)

    def charIndexesFromSentence(self, sentence):
        strindexes = []
        for word in sentence.split(' '):
            PAD_indexes = [self.tag_PADid for i in range(self.max_len_char - len(word))]
            indexes = [self.char2index[char] for char in word]
            indexes.extend(PAD_indexes)
            strindexes.append(indexes)
            
        return strindexes
    
    def sentenceFromCharIndexes(self, indexes):
        word_list=[]
        for w in indexes:
            word=''
            for c_idx in w:
                if c_idx == self.tag_PADid:
                    break
                word+=self.index2char[c_idx]
            if word == '':
                break
            word_list.append(word)
        return word_list
        
    def charTensorFromSentence(self, sentence, device='cpu'):
        PAD_indexes_char = [self.tag_PADid for i in range(self.max_len_char)]
        strindexes = self.charIndexesFromSentence(sentence)
        PAD_indexes = [PAD_indexes_char for i in range(self.max_len_word - len(strindexes))]
        strindexes.extend(PAD_indexes)
        return torch.tensor(strindexes, dtype=torch.long, device=device).view(self.max_len_word,self.max_len_char)

# def tensorsFromPair(pair):
#     input_tensor = engdict.tensorFromSentence(pair[0], MAX_LENGTH, device=device)
#     target_tensor = posdict.tensorFromSentence(pair[1], MAX_LENGTH, device=device)
#     return (input_tensor, target_tensor)




class Lang2:
    def __init__(self, name):
        self.name = name
        self.tag_PADid = ds.TOKKEN_PAD_IDX
        
        self.word2index = {}
        self.word2count = {}
        self.index2word = defaultTokken.copy() # {tag_UNK: "<UNK>", tag_PAD: "<PAD>"}
        self.n_words = len(self.index2word) # 2  # UNK 와 PAD 포함
        
        self.char2index = {}
        self.char2count = {}
        self.index2char = defaultTokken.copy() # {tag_UNK: "<UNK>", tag_PAD: "<PAD>"}
        self.n_chars = len(self.index2char) # 2  # UNK 와 PAD 포함
        
    def addSentence(self, sentence: list):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
        for char in word:
            self.addChar(char)
            
    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1
    
    def indexesFromSentence(self, sentence: list) -> list:
        return [self.word2index[word] for word in sentence]

    def sentenceFromIndexes(self, indexes) -> list:
        return [self.index2word[idx] for idx in indexes]

    def batch2tensor(self, batch: list, device='cpu') -> torch.tensor:
        max_len = max([len(s) for s in batch])
        fortensor = []
        for sentence in batch:
            indexes = self.indexesFromSentence(sentence)
            PAD_indexes = [self.tag_PADid for i in range(max_len - len(indexes))]
            fortensor.append(indexes.extend(PAD_indexes))
        return torch.tensor(fortensor, dtype=torch.long, device=device)
    
    
    
#     def tensorFromSentence(self, sentence, device='cpu'):
#         indexes = self.indexesFromSentence(sentence)
#         PAD_indexes = [self.tag_PADid for i in range(self.max_len_word - len(indexes))]
#         indexes.extend(PAD_indexes)
#         return torch.tensor(indexes, dtype=torch.long, device=device).view(self.max_len_word)

    def charIndexesFromSentence(self, sentence):
        strindexes = []
        for word in sentence.split(' '):
            PAD_indexes = [self.tag_PADid for i in range(self.max_len_char - len(word))]
            indexes = [self.char2index[char] for char in word]
            indexes.extend(PAD_indexes)
            strindexes.append(indexes)
            
        return strindexes
    
    def sentenceFromCharIndexes(self, indexes):
        word_list=[]
        for w in indexes:
            word=''
            for c_idx in w:
                if c_idx == self.tag_PADid:
                    break
                word+=self.index2char[c_idx]
            if word == '':
                break
            word_list.append(word)
        return word_list
        
    def charTensorFromSentence(self, sentence, device='cpu'):
        PAD_indexes_char = [self.tag_PADid for i in range(self.max_len_char)]
        strindexes = self.charIndexesFromSentence(sentence)
        PAD_indexes = [PAD_indexes_char for i in range(self.max_len_word - len(strindexes))]
        strindexes.extend(PAD_indexes)
        return torch.tensor(strindexes, dtype=torch.long, device=device).view(self.max_len_word,self.max_len_char)
 