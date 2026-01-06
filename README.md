🚗 Caption Recommendation System for Automotive Digital Marketing

Sistem ini merupakan Sistem Rekomendasi Caption Media Sosial yang dirancang khusus untuk kebutuhan digital marketing di bidang otomotif, seperti promosi produk mobil, branding brand otomotif, event auto show, dan kampanye media sosial berbagai brand kendaraan.

Aplikasi ini dibangun menggunakan pendekatan Content-Based Filtering dengan TF-IDF dan Cosine Similarity, serta mempertimbangkan kualitas konten berdasarkan engagement untuk menghasilkan rekomendasi caption yang relevan dan berkinerja baik.

🔗 Live App (Streamlit Cloud):
👉 ([link Streamlit](https://caption-recommendation-system-3yukqfvkt5pgkvhrsqevep.streamlit.app/))



📌 Fitur Utama
- 🔍 Rekomendasi caption berdasarkan kemiripan teks
- 🎯 Filter berdasarkan brand dan platform media sosial
- 📊 Penggabungan similarity score dan quality score
- ⚙️ Jumlah rekomendasi dapat diatur (1–10)
- 🚀 Deploy online menggunakan Streamlit Cloud


🧠 Metodologi Sistem
	Sistem ini menggunakan pendekatan Content-Based Recommendation System dengan tahapan sebagai berikut:
	1. Text Preprocessing
		Caption input dan caption dataset diproses melalui tahapan:
		- Case folding
		- Cleaning (remove punctuation, angka, dll)
		- Tokenization
		- Stopword removal
		- Normalisasi teks
	2. Text Representation (TF-IDF)
		Caption direpresentasikan dalam bentuk vektor numerik menggunakan Term Frequency–Inverse Document Frequency (TF-IDF).
	3. Similarity Calculation (Cosine Similarity)
		Tingkat kemiripan antara caption input dan caption dataset dihitung menggunakan Cosine Similarity.
	4. Quality Score (Engagement-Based)
		Sistem mempertimbangkan kualitas konten berdasarkan performa engagement (misalnya like, comment, atau metrik terkait).
	5. Final Scoring
		Skor akhir rekomendasi dihitung dari kombinasi:
				final_score = (α × similarity_score) + (β × quality_score)
		di mana:
		- similarity_score menunjukkan relevansi teks
		- quality_score menunjukkan kualitas performa konten

🗂️ Struktur Folder
caption-recommendation-system/
│
├── app.py                     # Main Streamlit app
├── config.py                  # Konfigurasi aplikasi
├── requirements.txt           # Dependency list
│
├── data/
│   └── captions_final.parquet # Dataset caption otomotif
│
├── logic/
│   ├── __init__.py
│   ├── preprocessing.py       # Text preprocessing
│   ├── recommender.py         # Logic rekomendasi
│   └── scoring.py             # Perhitungan quality & final score
│
├── models/
│   └── tfidf_vectorizer.pkl   # TF-IDF Vectorizer
│
└── README.md

▶️ Cara Menjalankan Secara Lokal
1️⃣ Clone Repository
git clone https://github.com/fetakalihwigati/caption-recommendation-system.git
cd caption-recommendation-system

2️⃣ Buat Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac / Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Jalankan Aplikasi
streamlit run app.py


Aplikasi akan berjalan di:

http://localhost:8501

🌐 Deployment

Aplikasi ini dideploy menggunakan Streamlit Community Cloud dengan integrasi langsung ke repository GitHub.

🎯 Ruang Lingkup Sistem

Sistem ini hanya menyajikan konten dalam bidang otomotif, khusus untuk keperluan digital marketing, meliputi:

Caption media sosial brand mobil

Promosi produk kendaraan

Kampanye event otomotif (misalnya auto show)

Aktivitas branding produk dan brand otomotif

📚 Teknologi yang Digunakan

Python

Pandas

Scikit-learn

Streamlit

Joblib

TF-IDF & Cosine Similarity

👤 Author

Feta Kalih Wigati
Program Studi Data Science
Bina Nusantara University

⭐ Catatan

Proyek ini dikembangkan sebagai bagian dari pembelajaran dan implementasi Natural Language Processing (NLP) dan Sistem Rekomendasi untuk kebutuhan digital marketing.
