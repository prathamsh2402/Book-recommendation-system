import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- Load Data ----
@st.cache_data
def load_data():
    books = pd.read_csv(r"C:\Projects\AI_PROJECT\Csv files\Books.csv.zip")
    users = pd.read_csv(r"C:\Projects\AI_PROJECT\Csv files\Users.csv.zip")
    ratings = pd.read_csv(r"C:\Projects\AI_PROJECT\Csv files\Ratings.csv.zip")

    return books, users, ratings

books, users, ratings = load_data()

# ---- Data Preprocessing ----
# Merge ratings with books to get useful data
book_data = ratings.merge(books, on='ISBN')

# Compute average rating and number of ratings
book_stats = book_data.groupby('Book-Title').agg({'Book-Rating': ['mean', 'count']})
book_stats.columns = ['Average-Rating', 'Num-Ratings']
book_stats = book_stats.reset_index()

# Merge stats back
books_final = pd.merge(books, book_stats, on='Book-Title', how='left')

# ---- Create TF-IDF based text representation ----
books_final['Combined'] = books_final['Book-Title'].fillna('') + ' ' + \
                          books_final['Book-Author'].fillna('') + ' ' + \
                          books_final['Publisher'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_final['Combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---- Recommend Function ----
def recommend(book_title, top_n=5):
    if book_title not in books_final['Book-Title'].values:
        return []
    idx = books_final[books_final['Book-Title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return books_final.iloc[book_indices][['Book-Title', 'Book-Author', 'Average-Rating']]

# ---- Streamlit UI ----
st.title("📚 Book Recommendation System (Cosine Similarity Based)")
st.write("Get book recommendations based on similar titles, authors, and publishers!")

selected_book = st.selectbox(
    "Select or Type a Book Title",
    options=books_final['Book-Title'].dropna().unique()
)

if st.button("Recommend"):
    recommendations = recommend(selected_book)
    if len(recommendations) == 0:
        st.warning("Book not found in dataset.")
    else:
        st.subheader("Top Recommendations:")
        for i, row in recommendations.iterrows():
            st.markdown(f"**{row['Book-Title']}**  \nAuthor: *{row['Book-Author']}*  \n⭐ Avg Rating: {round(row['Average-Rating'],2)}")
            st.write("---")
