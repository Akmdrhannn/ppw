import streamlit as st
import re
from streamlit_option_menu import option_menu
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
# Download kamus stop words
nltk.download('stopwords')
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans  
from sklearn.tree import DecisionTreeClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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
    # st.subheader("Term Frequency")
    # termfrequency = pd.read_csv('data/TermFrequensi.csv')
    # termfrequency
    # st.subheader("Logarithm Frequency")
    # logfrequency = pd.read_csv('data/Logarithm_Frequensi.csv')
    # logfrequency
    # st.subheader("TF-IDF")
    # tfidf = pd.read_csv('data/TF-IDF.csv')
    # tfidf



elif selected_option == "Modelling dan Reduksi Dimensi":
    st.header("Modelling")
    st.subheader("LDA Topic Modelling")
    st.write("""
        note : \n
            Label 0 = RPL\n
            Label 1 = KK
    """)

    st.write("LDA memungkinkan kita untuk memahami topik-topik umum yang muncul dalam koleksi dokumen tanpa memerlukan label atau anotasi topik dari manusia karena diambil menggunakan probabilitas kata kata dalam setiap topik")
    tflda = pd.read_csv('data/TermFrequensi.csv')
    st.write("Data awal import seperti ini",tflda)

    kelas_dataset = tflda['Label']
    
    # Ubah kelas RPL menjadi 0 dan kelas KK menjadi 1
    kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in kelas_dataset]
    # Contoh cetak hasilnya
    tflda['Label']=kelas_dataset_binary
    y = tflda['Label']
    # st.write("Lalu diubah kelas/label menjadi 0 untuk RPL dan 1 untuk KK",kelas_dataset,y)


    X = tflda.drop("Dokumen",axis=1)
    vectorizer = CountVectorizer(stop_words='english')
    X =vectorizer.fit_transform(tflda['Dokumen'].values.astype('U'))
    # st.write(X)


    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #LDA
    k = 3
    alpha = 0.1
    beta = 0.2

    lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    # Latih model LDA pada training set yg sudah di vectorisasi
    proporsi_topik_dokumen_train = lda.fit_transform(X_train)
    # Proyeksi dokumen pada test set ke topik yg telah dipelajari
    proporsi_topik_dokumen_test = lda.transform(X_test)
#     st.write("Lalu data dipisahkan atau di split")
#     st.code("""
#     #split data
#     y = tflda['Label']
#     X = tflda.drop("Dokumen",axis=1)
#     vectorizer = CountVectorizer(stop_words='english')
#     X =vectorizer.fit_transform(tflda['Dokumen'].values.astype('U'))
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
# """)
#     st.write("Setelah itu dilakukan LDA beserta pelatihan")
#     st.code(""""
            
#     #LDA
#     k = 3
#     alpha = 0.1
#     beta = 0.2

#     lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
#             """)
    
#     st.code(""""
#     # Latih model LDA pada training set yg sudah di vectorisasi
#     proporsi_topik_dokumen_train = lda.fit_transform(X_train)
#     # Proyeksi dokumen pada test set ke topik yg telah dipelajari
#     proporsi_topik_dokumen_test = lda.transform(X_test)
#             """)

    # proporsi topik
    dokumen = tflda['Dokumen']
    label= tflda['Label']
    output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen_test, columns=['Topik 1', 'Topik 2', 'Topik 3'])
    output_proporsi_TD.insert(0,'Dokumen', dokumen)
    output_proporsi_TD.insert(len(output_proporsi_TD.columns),'Label', tflda['Label'])
    csvawal = pd.read_csv('data/data_label_crawling.csv')
    judul = csvawal['Judul']
    output_judul = pd.concat([judul,output_proporsi_TD],axis=1)
    st.write("Data setelah di LDA kan")
    output_judul

    # Output distribusi
    # st.subheader("Output distribusi kata pada topik")
    # distribusi_kata_topik = pd.DataFrame(lda.components_)
    # distribusi_kata_topik


    #LDA kmeans
    st.subheader("LDA Kmeans")
    X_clustering = proporsi_topik_dokumen_test
    n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_clustering)
    
    # Menggabungkan hasil clustering ke DataFrame hasil LDA
    output_proporsi_TD['Cluster'] = clusters
    # Menggabungkan DataFrame hasil LDA dan DataFrame hasil clustering
    st.write("Menggabungkan dataframe LDA dan Cluster")
    output_gabung = pd.concat([output_proporsi_TD], axis=1)
    output_gabung['Label'] = output_gabung['Label'].replace({0: 'RPL', 1: 'KK'})
    
    # Menampilkan hasil akhir
    output_judul = pd.concat([judul, output_gabung], axis=1)
    output_judul




    # Naive Bayes
    st.subheader("Perhitungan akurasi Naive Bayes, KNN dan Decision Tree")

    # # Membuat model Naive Bayes
    # naive_bayes = GaussianNB()
    # naive_bayes.fit(X_train.toarray(), y_train)
    # predictions = naive_bayes.predict(X_test.toarray())
    # accuracy = round(accuracy_score(y_test, predictions)*100,2)
    # accnb = round(naive_bayes.score(X_train,y_train)*100,2)
    
    # st.write("Akurasi Naive Bayes:", accuracy)

    # Membuat model Naive Bayes
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    predictions = naive_bayes.predict(X_test)
    accuracy = round(accuracy_score(y_test, predictions)*100,2)
    accnb = round(naive_bayes.score(X_train,y_train)*100,2)
    
    st.write("Akurasi Naive Bayes:", accuracy)


    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train.toarray(),y_train)
    predict = knn.predict(X_test.toarray())
    accuracyknn = round(accuracy_score(y_test,predict)*100,2)
    accknn = round(knn.score(X_train,y_train)*100,2)

    st.write("Akurasi KNN :", accknn)
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    predictions_dt = decision_tree.predict(X_test)
    accuracy_dt = round(accuracy_score(y_test, predictions_dt) * 100, 2)
    acc_dt = round(decision_tree.score(X_train, y_train) * 100, 2)

    st.write("Akurasi Decision Tree:", accuracy_dt)


# Split sebelum LDA, Data X train,Y train digunakan
