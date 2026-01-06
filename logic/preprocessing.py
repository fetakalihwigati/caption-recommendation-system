# # def preprocess_text(text):
# #     text = str(text).lower()  # ubah ke huruf kecil
# #     text = re.sub(r'http\S+|www\S+', '', text)  # hapus URL
# #     text = re.sub(r'[^a-z\s]', ' ', text)  # hapus angka & simbol
# #     text = re.sub(r'\s+', ' ', text).strip()  # rapikan spasi

# #     tokens = word_tokenize(text)  # tokenisasi
# #     tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # hapus stopword & token pendek

# #     processed_tokens = []
# #     for word in tokens:
# #         if word.endswith(('nya', 'kan', 'lah', 'ku', 'mu')) or word in ['dan', 'yang', 'untuk', 'dengan']:
# #             word = stemmer_id.stem(word)  # stemming Indonesia
# #         else:
# #             word = stemmer_en.stem(word)  # stemming Inggris
# #         processed_tokens.append(word)

# #     return ' '.join(processed_tokens)

# import re
# # from nltk.tokenize import word_tokenize
# from nltk.tokenize import TreebankWordTokenizer
# from nltk.corpus import stopwords
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from nltk.stem import PorterStemmer

# # stop_words = set(
# #     stopwords.words('indonesian') +
# #     stopwords.words('english')
# # )

# # stemmer_id = StemmerFactory().create_stemmer()
# # stemmer_en = PorterStemmer()

# tokenizer = TreebankWordTokenizer()

# STOPWORDS_ID = {
#     "dan", "yang", "untuk", "dengan", "dari", "di", "ke",
#     "pada", "ini", "itu", "adalah", "sebagai", "juga",
#     "akan", "karena", "atau", "oleh", "saat", "dalam"
# }

# STOPWORDS_EN = {
#     "the", "and", "for", "with", "from", "this", "that",
#     "is", "are", "to", "of", "in", "on", "it", "as"
# }

# stop_words = STOPWORDS_ID.union(STOPWORDS_EN)

# # pastikan ini sudah didefinisikan sebelumnya
# # stop_words
# # stemmer_id
# # stemmer_en

# def preprocess_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+', '', text)
#     text = re.sub(r'[^a-z\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()

#     # tokens = word_tokenize(text)
#     tokens = tokenizer.tokenize(text)
#     tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

#     processed_tokens = []
#     for word in tokens:
#         if word.endswith(('nya', 'kan', 'lah', 'ku', 'mu')) or word in ['dan', 'yang', 'untuk', 'dengan']:
#             word = stemmer_id.stem(word)
#         else:
#             word = stemmer_en.stem(word)
#         processed_tokens.append(word)

#     return ' '.join(processed_tokens)


# # ✅ ALIAS UNTUK PRODUKSI
# def clean_text(text):
#     return preprocess_text(text)

# logic/preprocessing.py

import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# =====================
# INIT NLP TOOLS (GLOBAL)
# =====================
stemmer_id = StemmerFactory().create_stemmer()
stopwords_id = set(
    StopWordRemoverFactory().get_stop_words()
)


# =====================
# TEXT PREPROCESSING
# =====================
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove url
    text = re.sub(r"http\S+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # tokenize
    words = text.split()

    # stopword removal + stemming
    processed_words = [
        stemmer_id.stem(word)
        for word in words
        if word not in stopwords_id
    ]

    return " ".join(processed_words)


def clean_text(text: str) -> str:
    return preprocess_text(text)
