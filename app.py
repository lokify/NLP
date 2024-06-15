import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
import docx
import base64
from io import BytesIO

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')


# Function to preprocess text using nltk
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)


# Function to extract text from PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ''
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text


# Function to extract text from Word document
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + ' '
    return text


# Function to calculate TF-IDF vectors
def vectorize_documents(doc1, doc2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2]).toarray()
    return vectors


# Function to calculate Cosine Similarity
def calculate_cosine_similarity(vector1, vector2):
    cos_sim = cosine_similarity([vector1], [vector2])
    return cos_sim[0][0]


# Function to calculate Euclidean Similarity
def calculate_euclidean_similarity(vector1, vector2):
    euclidean_dist = euclidean(vector1, vector2)
    max_dist = np.linalg.norm(np.ones_like(vector1))
    return 1 - (euclidean_dist / max_dist)


# Function to calculate Jaccard Similarity
def calculate_jaccard_similarity(doc1, doc2):
    set1 = set(doc1.lower().split())
    set2 = set(doc2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# Function to generate a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
    return wordcloud


# Function to highlight differences
def highlight_differences(doc1, doc2):
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    common = set1 & set2
    unique1 = set1 - set2
    unique2 = set2 - set1
    return common, unique1, unique2


def get_wordcloud_download_link(wordcloud, filename="wordcloud.png"):
    """Generates a download link for a matplotlib wordcloud."""
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    # Convert plot to PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    # Generate download link
    b64 = base64.b64encode(img.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'


def main():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f0f0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📄 Document Similarity Checker with NLTK")
    st.markdown("""
    **Welcome to the Document Similarity Checker!** This tool helps you compare two text documents and provides similarity scores using different metrics.
    - **Cosine Similarity**: Measures the cosine of the angle between two vectors, capturing orientation and ignoring magnitude.
    - **Euclidean Similarity**: Measures the "straight-line" distance between vectors in space, adjusted to provide a similarity score.
    - **Jaccard Similarity**: Measures the overlap between sets of words in the documents, providing a ratio of common to total unique words.
    """)

    st.sidebar.title("Upload Documents")
    uploaded_file1 = st.sidebar.file_uploader("Choose Document 1", type=["txt", "pdf", "docx"])
    uploaded_file2 = st.sidebar.file_uploader("Choose Document 2", type=["txt", "pdf", "docx"])

    show_details = st.sidebar.checkbox("Show Detailed Results", False)

    if uploaded_file1 and uploaded_file2:
        file_type1 = uploaded_file1.name.split(".")[-1]
        file_type2 = uploaded_file2.name.split(".")[-1]

        if file_type1 == 'pdf':
            doc1 = extract_text_from_pdf(uploaded_file1)
        elif file_type1 == 'docx':
            doc1 = extract_text_from_docx(uploaded_file1)
        else:
            doc1 = uploaded_file1.read().decode("utf-8")

        if file_type2 == 'pdf':
            doc2 = extract_text_from_pdf(uploaded_file2)
        elif file_type2 == 'docx':
            doc2 = extract_text_from_docx(uploaded_file2)
        else:
            doc2 = uploaded_file2.read().decode("utf-8")
    else:
        doc1 = st.text_area("Enter Document 1")
        doc2 = st.text_area("Enter Document 2")

    if st.button("Calculate Similarities"):
        if doc1 and doc2:
            # Preprocess documents
            doc1_preprocessed = preprocess(doc1)
            doc2_preprocessed = preprocess(doc2)

            # Vectorize documents
            vectors = vectorize_documents(doc1_preprocessed, doc2_preprocessed)
            vector1, vector2 = vectors[0], vectors[1]

            # Calculate similarities
            cosine_sim = calculate_cosine_similarity(vector1, vector2)
            euclidean_sim = calculate_euclidean_similarity(vector1, vector2)
            jaccard_sim = calculate_jaccard_similarity(doc1_preprocessed, doc2_preprocessed)

            # Calculate overall similarity
            overall_similarity = np.mean([cosine_sim, euclidean_sim, jaccard_sim])

            # Determine similarity level
            if overall_similarity > 0.75:
                similarity_statement = "very similar"
            elif overall_similarity > 0.5:
                similarity_statement = "somewhat similar"
            else:
                similarity_statement = "not very similar"

            # Display results with scores
            st.subheader("Similarity Scores")

            if show_details:
                st.markdown(f"**Cosine Similarity:** {cosine_sim:.2f}")
                st.progress(cosine_sim)
                st.markdown(f"**Euclidean Similarity:** {euclidean_sim:.2f}")
                st.progress(euclidean_sim)
                st.markdown(f"**Jaccard Similarity:** {jaccard_sim:.2f}")
                st.progress(jaccard_sim)
            else:
                st.markdown(f"**Cosine Similarity:** {cosine_sim:.2f}")
                st.markdown(f"**Euclidean Similarity:** {euclidean_sim:.2f}")
                st.markdown(f"**Jaccard Similarity:** {jaccard_sim:.2f}")

            st.subheader("Overall Similarity")
            st.markdown(f"The documents are **{similarity_statement}** with an overall similarity score of {overall_similarity:.2f}.")

            # Display word clouds
            st.subheader("Word Clouds")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Document 1**")
                wordcloud1 = generate_wordcloud(doc1_preprocessed)
                plt.imshow(wordcloud1, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())
                st.markdown(get_wordcloud_download_link(wordcloud1, "wordcloud1.png"), unsafe_allow_html=True)
            with col2:
                st.write("**Document 2**")
                wordcloud2 = generate_wordcloud(doc2_preprocessed)
                plt.imshow(wordcloud2, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())
                st.markdown(get_wordcloud_download_link(wordcloud2, "wordcloud2.png"), unsafe_allow_html=True)

            # Highlight differences
            st.subheader("Differences")
            common, unique1, unique2 = highlight_differences(doc1_preprocessed, doc2_preprocessed)
            st.write("**Common Words:**", ', '.join(common))
            st.write("**Unique to Document 1:**", ', '.join(unique1))
            st.write("**Unique to Document 2:**", ', '.join(unique2))

            # Downloadable link for similarity scores
            csv = f"Document 1,Document 2\nCosine Similarity,{cosine_sim:.2f}\nEuclidean Similarity,{euclidean_sim:.2f}\nJaccard Similarity,{jaccard_sim:.2f}\nOverall Similarity,{overall_similarity:.2f}"
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown("#### Download Similarity Scores:")
            href = f'<a href="data:file/csv;base64,{b64}" download="similarity_scores.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.warning("Please enter both documents or upload text files.")


if __name__ == "__main__":
    main()
