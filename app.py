# # import streamlit as st
# # import pandas as pd
# # import pickle
# # import numpy as np

# # from sklearn.metrics.pairwise import cosine_similarity

# # # ======================================
# # # LOAD DATA & MODEL
# # # ======================================

# # @st.cache_data
# # def load_data():
# #     return pd.read_parquet("data/captions_final.parquet")

# # @st.cache_resource
# # def load_model():
# #     with open("models/tfidf_vectorizer.pkl", "rb") as f:
# #         vectorizer = pickle.load(f)

# #     tfidf_matrix = np.load("models/tfidf_matrix.npz")["arr_0"]
# #     return vectorizer, tfidf_matrix


# # df = load_data()
# # tfidf_vectorizer, tfidf_matrix = load_model()

# # # ======================================
# # # UI
# # # ======================================

# # st.set_page_config(page_title="Caption Recommender", layout="wide")

# # st.title("🚗 Sistem Rekomendasi Caption Otomotif")
# # st.write("Masukkan caption dan pilih tema untuk mendapatkan rekomendasi terbaik.")

# import streamlit as st

# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Otomotif",
#     layout="wide"
# )

# # =====================
# # HEADER
# # =====================
# st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# st.caption("Content-Based Filtering • TF-IDF • Cosine Similarity")

# st.divider()

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=1,
#     max_value=10,
#     value=5
# )

# brand_filter = st.sidebar.selectbox(
#     "Filter Brand (opsional)",
#     ["Semua", "Toyota", "Honda", "BYD", "KIA"]
# )

# # =====================
# # MAIN INPUT
# # =====================
# st.subheader("📝 Masukkan Caption")
# user_caption = st.text_area(
#     "Caption input",
#     placeholder="Contoh: Rasakan kenyamanan berkendara bersama mobil listrik terbaru..."
# )

# # =====================
# # ACTION BUTTON
# # =====================
# if st.button("🔍 Rekomendasikan Caption"):
#     if user_caption.strip() == "":
#         st.warning("⚠️ Caption tidak boleh kosong")
#     else:
#         st.info("🚧 Logic rekomendasi akan diproses di tahap berikutnya")

# st.divider()

# # =====================
# # OUTPUT PLACEHOLDER
# # =====================
# st.subheader("📌 Hasil Rekomendasi")
# st.write("Hasil rekomendasi akan muncul di sini.")

# from logic.recommender import recommend_caption

# if st.button("🔍 Rekomendasikan Caption"):
#     result = recommend_caption(
#         caption_input,
#         top_k=top_k,
#         brand=selected_brand
#     )

#     st.dataframe(result)



#Sudah BISA tidak error
# import streamlit as st
# import pandas as pd

# from logic.recommender import recommend_caption
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # from logic.preprocessing import preprocess_text
# # from logic.recommender import recommend_caption


# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Digital Marketing",
#     layout="wide"
# )

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=3,
#     max_value=10,
#     value=5
# )

# brand_filter = st.sidebar.selectbox(
#     "Filter Brand (opsional)",
#     ["Semua", 
#      "BYD", 
#      "Chery", 
#      "Citroen", 
#      "Daihatsu", 
#      "DFSK",
#      "Honda", 
#      "Hyundai", 
#      "KIA", 
#      "Mazda", 
#      "MG",
#      "Mitsubishi", 
#      "Neta", 
#      "Suzuki", 
#      "Toyota", 
#      "Wuling"]
# )


# # # =====================
# # # MAIN UI
# # # =====================
# # st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# # st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# # st.divider()

# # st.subheader("📝 Masukkan Caption")
# # caption_input = st.text_area(
# #     "Caption input",
# #     placeholder="Contoh: Rasakan kenyamanan berkendara bersama mobil listrik terbaru..."
# # )

# # if st.button("🔍 Rekomendasikan Caption"):
# #     if caption_input.strip() == "":
# #         st.warning("Masukkan caption terlebih dahulu")
# #     else:
# #         with st.spinner("Mencari caption terbaik..."):
# #             result = recommend_caption(
# #                 caption_input,
# #                 top_k=top_k,
# #                 brand=None
# #             )

# #         st.subheader("📌 Hasil Rekomendasi")
# #         st.dataframe(result)


# # =====================
# # MAIN UI
# # =====================
# st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# st.divider()

# # =====================
# # ABOUT APP
# # =====================
# with st.expander("ℹ️ Tentang Aplikasi"):
#     st.write(
#         """
#         Aplikasi ini bertujuan untuk membantu pengguna menemukan caption digital marketing
#         yang relevan berdasarkan **kemiripan konten teks** dan **kualitas caption**.

#         Sistem menggunakan pendekatan **Content-Based Filtering**
#         dengan representasi teks **TF-IDF** dan perhitungan **Cosine Similarity**.
#         """
#     )

# # =====================
# # INPUT
# # =====================
# st.subheader("📝 Masukkan Caption")
# caption_input = st.text_area(
#     "Caption input",
#     placeholder="Contoh: Rasakan kenyamanan berkendara bersama mobil listrik terbaru..."
# )

# st.caption(
#     "Tema konten digunakan untuk memfokuskan rekomendasi sesuai tujuan komunikasi."
# )

# # =====================
# # ACTION
# # =====================
# if st.button("🔍 Rekomendasikan Caption"):
#     if caption_input.strip() == "":
#         st.warning("Masukkan caption terlebih dahulu.")
#     else:
#         with st.spinner("Mencari caption terbaik..."):
#             result = recommend_caption(
#                 input_caption=caption_input,
#                 top_k=top_k,
#                 brand=None
#             )

#         if result.empty:
#             st.warning(
#                 "Tidak ditemukan caption yang sesuai dengan brand yang dipilih. "
#                 "Coba pilih brand 'Semua' atau gunakan caption lain."
#             )
#         else:
#             st.subheader("📌 Hasil Rekomendasi")
#             st.dataframe(result, use_container_width=True)


# import streamlit as st
# import pandas as pd

# from logic.recommender import recommend_caption

# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Digital Marketing",
#     layout="wide"
# )

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=3,
#     max_value=10,
#     value=5
# )

# meta_theme = st.sidebar.selectbox(
#     "Tema Konten",
#     [
#         "Semua",
#         "Product",
#         "Branding",
#         "Interactive",
#         "Event",
#         "Community"
#     ]
# )

# # =====================
# # MAIN UI
# # =====================
# st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# st.divider()

# # =====================
# # ABOUT APP
# # =====================
# with st.expander("ℹ️ Tentang Aplikasi"):
#     st.write(
#         """
#         Aplikasi ini bertujuan untuk membantu pengguna menemukan caption digital marketing
#         yang relevan berdasarkan **kemiripan konten teks** dan **kualitas caption**.

#         Sistem menggunakan pendekatan **Content-Based Filtering**
#         dengan representasi teks **TF-IDF** dan perhitungan **Cosine Similarity**.
#         """
#     )

# # =====================
# # INPUT
# # =====================
# st.subheader("📝 Masukkan Caption")
# caption_input = st.text_area(
#     "Caption input",
#     placeholder="Contoh: Rasakan kenyamanan berkendara bersama mobil listrik terbaru..."
# )

# st.caption(
#     "Tema konten digunakan untuk memfokuskan rekomendasi sesuai tujuan komunikasi."
# )

# # =====================
# # ACTION
# # =====================
# if st.button("🔍 Rekomendasikan Caption"):
#     if caption_input.strip() == "":
#         st.warning("Masukkan caption terlebih dahulu.")
#     else:
#         with st.spinner("Mencari caption terbaik..."):
#             result = recommend_caption(
#                 input_caption=caption_input,
#                 top_k=top_k,
#                 meta_theme=meta_theme
#             )

#         if result.empty:
#             st.warning(
#                 "Tidak ditemukan caption yang sesuai dengan tema yang dipilih. "
#                 "Coba pilih tema 'Semua' atau gunakan caption lain."
#             )
#         else:
#             st.subheader("📌 Hasil Rekomendasi")
#             st.dataframe(result, use_container_width=True)

#terakhir 
# import streamlit as st
# import pandas as pd

# from logic.recommender import recommend_caption

# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Digital Marketing",
#     layout="wide"
# )

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=3,
#     max_value=10,
#     value=5
# )

# theme_option = st.sidebar.selectbox(
#     "Pilih Tema Caption",
#     [
#         "Semua",
#         "Product",
#         "Branding",
#         "Interactive",
#         "Event",
#         "Community"
#     ]
# )

# # =====================
# # MAIN UI
# # =====================
# st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# st.divider()

# st.subheader("📝 Masukkan Caption")
# caption_input = st.text_area(
#     "Caption input",
#     placeholder="Contoh: Rasakan pengalaman berkendara yang lebih modern dan nyaman..."
# )

# if st.button("🔍 Rekomendasikan Caption"):
#     if caption_input.strip() == "":
#         st.warning("Masukkan caption terlebih dahulu")
#     else:
#         with st.spinner("Mencari caption terbaik..."):
#             result = recommend_caption(
#                 input_caption=caption_input,
#                 top_k=top_k,
#                 meta_theme=theme_option
#             )

#         st.subheader("📌 Hasil Rekomendasi")

#         if result.empty:
#             st.info("Tidak ditemukan caption yang sesuai dengan filter tema.")
#         else:
#             st.dataframe(
#                 result,
#                 use_container_width=True
#             )


# import streamlit as st
# import pandas as pd

# from logic.recommender import recommend_caption

# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Digital Marketing",
#     layout="wide"
# )

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=3,
#     max_value=10,
#     value=5
# )

# brand_options = [
#     "byd", "chery", "citroen", "daihatsu", "dfsk",
#     "honda", "hyundai", "kia", "mazda", "mg",
#     "mitsubishi", "neta", "suzuki", "toyota", "wuling"
# ]

# selected_brands = st.sidebar.multiselect(
#     "Pilih Brand (opsional)",
#     options=brand_options,
#     default=[]
# )

# # tema_options = [
# #     "Product", "Branding", "Interactive", "Event", "Community"
# # ]

# # selected_brands = st.sidebar.multiselect(
# #     "Pilih Tema (opsional)",
# #     options=tema_options,
# #     default=[]
# # )

# # =====================
# # MAIN UI
# # =====================
# st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# st.divider()

# with st.expander("ℹ️ Tentang Aplikasi", expanded=True):
#     st.write(
#         """
#         Aplikasi ini membantu menemukan caption digital marketing yang relevan
#         berdasarkan **kemiripan teks** dan **kualitas caption**.

#         Sistem menggunakan pendekatan **Content-Based Filtering**
#         dengan **TF-IDF** dan **Cosine Similarity**.
#         """
#     )

# st.subheader("📝 Masukkan Caption")
# caption_input = st.text_area(
#     "Caption input",
#     placeholder="Contoh: Rasakan kenyamanan berkendara bersama mobil listrik terbaru..."
# )

# if st.button("🔍 Rekomendasikan Caption"):
#     if caption_input.strip() == "":
#         st.warning("Masukkan caption terlebih dahulu.")
#     else:
#         with st.spinner("Mencari caption terbaik..."):
#             result = recommend_caption(
#                 input_caption=caption_input,
#                 top_k=top_k,
#                 brands=selected_brands
#             )

#         if result.empty:
#             st.warning(
#                 "Tidak ditemukan caption untuk brand yang dipilih. "
#                 "Coba kosongkan filter brand."
#             )
#         else:
#             st.subheader("📌 Hasil Rekomendasi")
#             st.dataframe(
#                 result,
#                 use_container_width=True
#             )

#terbaru
# import streamlit as st
# import pandas as pd

# from logic.recommender import recommend_caption
# from logic.recommender import load_assets


# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Digital Marketing",
#     layout="wide"
# )

# # =====================
# # LOAD DATA (UNTUK DROPDOWN)
# # =====================
# df, _, _ = load_assets()


# # =====================
# # HEADER
# # =====================
# st.title("Sistem Rekomendasi Caption Media Sosial untuk Digital Marketing Otomotif Mobil")
# st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# st.divider()

# # =====================
# # ABOUT APP
# # =====================
# st.markdown("""
# ### 🚗 Tentang Sistem Rekomendasi Caption Konten Media Sosial 

# Sistem ini akan merekomendasikan **caption digital marketing** yang paling relevan berdasarkan kemiripan 
# **teks caption input** dan **karakteristik konten sebelumnya** dengan menggunakan pendekatan 
# **Content-Based Filtering** dengan **TF-IDF** dan **Cosine Similarity**, serta mempertimbangkan 
# **kualitas konten** melalui **quality score** berbasis engagement untuk menghasilkan rekomendasi yang 
# relevan dan memiliki performa yang baik.

# Sistem rekomendasi ini dirancang secara khusus untuk menyajikan **konten caption media sosial dalam bidang 
# otomotif** yang digunakan pada kebutuhan **digital marketing.**
# Konten yang direkomendasikan mencakup caption untuk **media sosial berbagai brand mobil, promosi produk 
# kendaraan, kampanye event otomotif (seperti auto show)**, serta aktivitas **branding produk dan brand otomotif.**

# """)

# st.markdown("""
# ### ⚙️ Metodologi Sistem

# 1. Caption input diproses melalui tahapan *text preprocessing*.
# 2. Teks direpresentasikan dalam bentuk vektor menggunakan **TF-IDF**.
# 3. Kemiripan dihitung menggunakan **Cosine Similarity**.
# 4. Skor akhir dihitung dari kombinasi:
#    - *Similarity Score*
#    - *Quality Score*.
# 5. Hasil diurutkan berdasarkan **Final Score** tertinggi.
# """)

# st.markdown("""
# ### 📖 Cara Menggunakan Sistem Rekomendasi Caption

# 1. **Pengguna memasukkan ide caption media sosial bertema otomotif mobil**
            
#    Pengguna menuliskan teks caption bertema otomotif mobil yang ingin dikembangkan atau dicari rekomendasinya oleh sistem.  
#    Contoh:
#    - *Datang segera ke booth Suzuki dan ikuti test drive gratis Suzuki Grand Vitara.*
#    - *Teman Wuling, yuk ikutan #WulingQuiz dan dapatkan e-wallet bagi kamu yang beruntung!*
#    - *Rasakan kenyamanan berkendara mobil listrik dengan fitur yang lebih modern.*

            
            
# 2. **Pengguna mengatur parameter rekomendasi**
            
#    Pengguna dapat menyesuaikan pengaturan sistem sesuai kebutuhan, meliputi:
#    - **Jumlah** caption **rekomendasi** yang diinginkan (3–10 caption)
#    - Filter **brand**
#    - Filter **platform** (Instagram, Facebook, TikTok, atau YouTube)

            

# 3. **Sistem menjalankan proses rekomendasi**
            
#    Sistem akan memproses caption input dan menghasilkan rekomendasi caption yang paling relevan  
#    berdasarkan kemiripan teks dan kualitas konten.
# """)


# st.divider()

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=3,
#     max_value=10,
#     value=5
# )

# brand_option = st.sidebar.selectbox(
#     "Pilih Brand",
#     ["Semua"] + sorted(df["brand"].dropna().unique().tolist())
# )

# platform_option = st.sidebar.selectbox(
#     "Pilih Platform",
#     ["Semua", "INSTAGRAM", "FACEBOOK", "TIKTOK"]
# )

# # =====================
# # INPUT SECTION
# # =====================
# st.subheader("📝 Masukkan Caption")

# caption_input = st.text_area(
#     "Caption Input",
#     placeholder="Contoh: Rasakan pengalaman berkendara yang lebih modern dan nyaman --ATAU-- Teman Wuling, yuk ikutan #WulingQuiz dan dapatkan E-wallet bagi kamu yang beruntung!"
# )

# # =====================
# # ACTION
# # =====================
# if st.button("🔍 Rekomendasikan Caption"):
#     if caption_input.strip() == "":
#         st.warning("Silakan masukkan caption terlebih dahulu.")
#     else:
#         with st.spinner("Mencari caption terbaik..."):
#             result = recommend_caption(
#                 input_caption=caption_input,
#                 top_k=top_k,
#                 brand=brand_option,
#                 platform=platform_option
#             )

#         st.subheader("📌 Hasil Rekomendasi")

#         if result.empty:
#             st.info("Tidak ditemukan caption yang sesuai dengan filter yang dipilih.")
#         else:
#             st.success("✅ Rekomendasi ditemukan!")
#             st.dataframe(
#                 result,
#                 use_container_width=True
#             )

import streamlit as st
import pandas as pd

from logic.recommender import recommend_caption, load_assets


# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Sistem Rekomendasi Caption Digital Marketing",
    layout="wide"
)

# =====================
# LOAD DATA (UNTUK DROPDOWN)
# =====================
df, _, _ = load_assets()


# =====================
# HEADER
# =====================
st.title("Sistem Rekomendasi Caption Media Sosial untuk Digital Marketing Otomotif Mobil")
st.caption("Content-Based Filtering · Word2Vec · Cosine Similarity · Engagement-based Ranking")

st.divider()

# =====================
# ABOUT APP
# =====================
st.markdown("""
### 🚗 Tentang Sistem Rekomendasi Caption Konten Media Sosial 

Sistem ini akan merekomendasikan **caption digital marketing** yang paling relevan berdasarkan kemiripan 
**teks caption input** dan **karakteristik konten sebelumnya** dengan menggunakan pendekatan 
**Content-Based Filtering** dengan **TF-IDF** dan **Cosine Similarity**, serta mempertimbangkan 
**kualitas konten** melalui **quality score** berbasis engagement untuk menghasilkan rekomendasi yang 
relevan dan memiliki performa yang baik.

Sistem rekomendasi ini dirancang secara khusus untuk menyajikan **konten caption media sosial dalam bidang 
otomotif** yang digunakan pada kebutuhan **digital marketing.**
Konten yang direkomendasikan mencakup caption untuk **media sosial berbagai brand mobil, promosi produk 
kendaraan, kampanye event otomotif (seperti auto show)**, serta aktivitas **branding produk dan brand otomotif.**

""")

st.markdown("""
### ⚙️ Metodologi Sistem

1. Caption input diproses melalui tahapan *text preprocessing*.
2. Teks direpresentasikan dalam bentuk vektor menggunakan **TF-IDF**.
3. Kemiripan dihitung menggunakan **Cosine Similarity**.
4. Skor akhir dihitung dari kombinasi:
   - *Similarity Score*
   - *Quality Score*.
5. Hasil diurutkan berdasarkan **Final Score** tertinggi.
""")

st.markdown("""
### 📖 Cara Menggunakan Sistem Rekomendasi Caption

1. **Pengguna memasukkan ide caption media sosial bertema otomotif mobil**
            
   Pengguna menuliskan teks caption bertema otomotif mobil yang ingin dikembangkan atau dicari rekomendasinya oleh sistem.  
   Contoh:
   - *Datang segera ke booth Suzuki dan ikuti test drive gratis Suzuki Grand Vitara.*
   - *Teman Wuling, yuk ikutan #WulingQuiz dan dapatkan e-wallet bagi kamu yang beruntung!*
   - *Rasakan kenyamanan berkendara mobil listrik dengan fitur yang lebih modern.*

            
            
2. **Pengguna mengatur parameter rekomendasi**
            
   Pengguna dapat menyesuaikan pengaturan sistem sesuai kebutuhan, meliputi:
   - **Jumlah** caption **rekomendasi** yang diinginkan (3–10 caption)
   - Filter **brand**
   - Filter **platform** (Instagram, Facebook, TikTok, atau YouTube)

            

3. **Sistem menjalankan proses rekomendasi**
            
   Sistem akan memproses caption input dan menghasilkan rekomendasi caption yang paling relevan  
   berdasarkan kemiripan teks dan kualitas konten.
""")


st.divider()

# =====================
# SIDEBAR
# =====================
st.sidebar.header("⚙️ Pengaturan Rekomendasi")

top_k = st.sidebar.slider(
    "Jumlah rekomendasi",
    min_value=3,
    max_value=10,
    value=5
)

brand_option = st.sidebar.selectbox(
    "Pilih Brand",
    ["Semua"] + sorted(df["brand"].dropna().unique().tolist())
)

platform_option = st.sidebar.selectbox(
    "Pilih Platform",
    ["Semua"] + sorted(df["platform"].dropna().unique().tolist())
)

meta_theme_option = st.sidebar.selectbox(
    "Pilih Meta Theme",
    ["Semua"] + sorted(df["meta_theme"].dropna().unique().tolist())
)

# =====================
# INPUT SECTION
# =====================
st.subheader("📝 Masukkan Ide Caption")

caption_input = st.text_area(
    "Caption Input",
    placeholder=(
        "Contoh:\n"
        "- Rasakan pengalaman berkendara yang lebih modern dan nyaman\n"
        "- Teman Wuling, yuk ikutan #WulingQuiz dan dapatkan E-wallet bagi kamu yang beruntung!"
    )
)

# =====================
# ACTION
# =====================
if st.button("🔍 Rekomendasikan Caption"):
    if caption_input.strip() == "":
        st.warning("Silakan masukkan caption terlebih dahulu.")
    else:
        with st.spinner("Mencari caption terbaik..."):
            result = recommend_caption(
                input_caption=caption_input,
                top_k=top_k,
                brand=brand_option,
                platform=platform_option,
                meta_theme=meta_theme_option
            )

        st.subheader("📌 Hasil Rekomendasi")

        if result.empty:
            st.info("Tidak ditemukan caption yang sesuai dengan filter yang dipilih.")
        else:
            st.success("✅ Rekomendasi ditemukan!")

            st.dataframe(
                result,
                use_container_width=True
            )


# import streamlit as st
# import pandas as pd

# from logic.recommender import recommend_caption

# # =====================
# # PAGE CONFIG
# # =====================
# st.set_page_config(
#     page_title="Sistem Rekomendasi Caption Digital Marketing",
#     layout="wide"
# )

# # =====================
# # SIDEBAR
# # =====================
# st.sidebar.header("⚙️ Pengaturan Rekomendasi")

# top_k = st.sidebar.slider(
#     "Jumlah rekomendasi",
#     min_value=3,
#     max_value=10,
#     value=5
# )

# brand_option = st.sidebar.selectbox(
#     "Pilih Brand",
#     [
#         "Semua",
#         "Toyota",
#         "Honda",
#         "Hyundai",
#         "Wuling",
#         "BYD",
#         "KIA"
#     ]
# )

# platform_option = st.sidebar.selectbox(
#     "Pilih Platform",
#     [
#         "Semua",
#         "INSTAGRAM",
#         "FACEBOOK",
#         "TIKTOK"
#     ]
# )

# # =====================
# # MAIN UI
# # =====================
# st.title("🚗 Sistem Rekomendasi Caption Digital Marketing")
# st.caption("Content-Based Filtering · TF-IDF · Cosine Similarity")

# st.divider()

# st.subheader("📝 Masukkan Caption")
# caption_input = st.text_area(
#     "Caption input",
#     placeholder="Contoh: Teman Wuling, yuk ikutan #WulingQuiz dan dapatkan E-wallet bagi kamu yang beruntung!"
# )

# if st.button("🔍 Rekomendasikan Caption"):
#     if caption_input.strip() == "":
#         st.warning("Masukkan caption terlebih dahulu")
#     else:
#         with st.spinner("Mencari caption terbaik..."):
#             result = recommend_caption(
#                 input_caption=caption_input,
#                 top_k=top_k,
#                 brand=brand_option,
#                 platform=platform_option
#             )

#         st.subheader("📌 Hasil Rekomendasi")

#         if result.empty:
#             st.info("Tidak ditemukan caption yang sesuai dengan filter brand / platform.")
#         else:
#             st.dataframe(
#                 result,
#                 use_container_width=True
#             )
