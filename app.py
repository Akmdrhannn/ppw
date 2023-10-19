import streamlit as st
import re
from streamlit_option_menu import option_menu
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
# Download kamus stop words
nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans  
from sklearn import tree
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Opsi menu tab di sidebar
with st.sidebar:
    selected_option = option_menu("Pilih Tab:", ["Home","Web Crawling", "Preprocessing", "Modelling dan Reduksi Dimensi"])

# Konten tab di halaman utama
if selected_option == "Home":
    st.header("PPW UTS")
    st.write("""
    Nama    : Akhmad Raihan Aulia Fikri\n
    Kelas   : PPW A\n
    NIM     : 200411100095

""")

elif selected_option == "Web Crawling":
    st.header("Web Crawling")
    st.write("Web crawling adalah proses di mana search engine menemukan konten yang di-update di sebuah situs atau halaman baru, perubahan situs, atau link yang mati")
    st.write("Data diambil dari PTA Trunojoyo prodi Teknik Informatika")
    webcrawl = pd.read_csv('data/crawling_pta.csv')
    webcrawl

elif selected_option == "Preprocessing":
    st.title("Preprocessing")
    st.write("Data Diberi Label")
    labelled = pd.read_csv('data/data_label_crawling.csv')
    labelled

    st.subheader("Setelah itu dilakukan cleaning data")
    st.write("""
            -Normalisasi\n
            -Tokenizing\n
            -Stopword\n
            -Stemming\n
             """)
    
    # normalisasi
    st.header("Normalisasi")
    # data null
    st.subheader("**Hapus data null**")
    st.write("Jumlah nilai null sebelum penghapusan:")
    st.write(labelled.isnull().sum())

    labelled_clean = labelled.dropna()

    st.write("Jumlah nilai null setelah penghapusan:")
    st.write(labelled_clean.isnull().sum())
    # karakter tertentu
    st.subheader("**Menghapus Karakter Spesial dan Angka**")
    def cleaning(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        return text

    # Membersihkan kolom 'Abstrak' dalam labelled_clean
    labelled_clean['Cleaning'] = labelled_clean['Abstrak'].apply(cleaning)

    # Menampilkan hasil pembersihan teks di Streamlit
    st.write("Hasil Pembersihan Teks")
    st.write(labelled_clean['Cleaning'])

    # cek karakter khusus
    st.subheader("Cek ulang karakter spesial")
    def cek_specialCharacter(dokumen):
        karakter = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/', '`', '~']
        special_characters = [char for char in dokumen if char in karakter]
        return special_characters

    # Memeriksa karakter khusus dalam teks bersih dan menampilkannya di Streamlit
    labelled_clean['Special Characters'] = labelled_clean['Cleaning'].apply(cek_specialCharacter)
    st.write(labelled_clean['Special Characters'])

    # tokenizing
    st.subheader("Tokenizing")
    st.write("Tokenizing adalah proses memecah teks atau dokumen menjadi potongan-potongan yang lebih kecil")
    def tokenizer(text):
        text = text.lower()
        return word_tokenize(text)

    labelled_clean['Tokenizing'] = labelled_clean['Cleaning'].apply(tokenizer)
    st.write(labelled_clean['Tokenizing'])

    st.write("Menghitung jumlah kata dari tokenisasi")
    def count_word(dokumens):
        return len(dokumens)

    labelled_clean['Count Word'] = labelled_clean['Tokenizing'].apply(count_word)
    labelled_clean

    # Stopword
    st.subheader("Stopword")
    st.write("Stopwords digunakan untuk menghilangkan kata umum yang sering muncul dalam teks seperti: di, dan, atau, dari, ke, saya.")
    corpus = stopwords.words('indonesian')
    def stopwordText(words):
        return [word for word in words if word not in corpus]

    labelled_clean['Stopword Removal'] = labelled_clean['Tokenizing'].apply(stopwordText)

    # Gabungkan kembali token menjadi kalimat utuh
    labelled_clean['Full Text'] = labelled_clean['Stopword Removal'].apply(lambda x: ' '.join(x))
    labelled_clean['Full Text']

    # Stemming
    st.subheader("Stemming")
    st.write("Stemming adalah proses normalisasi data teks menjadikan kata dasar")
    stemming = pd.read_csv('data/hasil_stemming.csv')
    stemming
    # def stemmingText(dokumens):
    #     factory = StemmerFactory()
    #     stemmer = factory.create_stemmer()

    #     return [stemmer.stem(i) for i in dokumens]

    # labelled_clean['Stemming'] = labelled_clean['Stopword Removal'].apply(stemmingText)
    # st.write(labelled_clean['Stemming'])

    # -------------------------------------
    st.subheader("Sehingga dari cleaning data diatas, dihasilkan beberapa hasil preprocessing seperti One Hot Encoding,TF IDF, Term Frequency, Logarithm Frequency")
    st.write('data tidak ditampilkan karna running time lama')

    # st.subheader("One Hot Encoding")
    # st.write("")
    # onehot = pd.read_csv('data/OneHotEncoder.csv')
    # onehot
    # st.subheader("TF-IDF")
    # tfidf = pd.read_csv('data/TF-IDF.csv')
    # tfidf
    # st.subheader("Term Frequency")
    # termfrequency = pd.read_csv('data/TermFrequensi.csv')
    # termfrequency
    # st.subheader("Logarithm Frequency")
    # logfrequency = pd.read_csv('data/Logarithm_Frequensi.csv')
    # logfrequency



elif selected_option == "Modelling dan Reduksi Dimensi":
    st.header("Modelling")
    st.subheader("LDA Topic Modelling")
    
    tflda = pd.read_csv('data/TermFrequensi.csv')
    tflda
