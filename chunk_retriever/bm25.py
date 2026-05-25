import math
from collections import Counter

class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        df = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            
            for word in frequencies:
                if word not in df:
                    df[word] = 0
                df[word] += 1
                
        for word, freq in df.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        for index in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_len[index]
            frequencies = self.doc_freqs[index]
            for word in query:
                if word not in frequencies:
                    continue
                freq = frequencies[word]
                numerator = self.idf[word] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            scores[index] = score
        return scores
