#!/usr/bin/env python
# coding: utf-8

# In[25]:


from collections import Counter
import streamlit as st
import pandas as pd
# import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import io
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import re
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load("it_core_news_sm")
italian_stopwords = nlp.Defaults.stop_words


# In[20]:


st.set_page_config(page_title='Clever analysis',  layout='wide', page_icon=':rocket:')


# In[21]:


# Set up RoBERTa sentiment pipeline
# roberta_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
st.title("Sentiment Analysis Tool")


# In[22]:


# Sidebar: Data Input
st.sidebar.image("logo.png", width=380)  # Adjust width if needed
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Select input source:", ["Upload CSV", "Use Twitter Handle", "Reddit Subreddit"])


# In[23]:


# Load data
df = pd.DataFrame()

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'text' and optional 'date' column:", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

# elif data_source == "Use Twitter Handle":
#     try:
#         import tweepy
#         st.sidebar.markdown("**Twitter API Credentials**")
#         api_key = st.sidebar.text_input("API Key")
#         api_secret = st.sidebar.text_input("API Secret", type="password")
#         access_token = st.sidebar.text_input("Access Token")
#         access_secret = st.sidebar.text_input("Access Token Secret", type="password")

#         handle = st.sidebar.text_input("Enter Twitter handle (without @):")
#         tweet_limit = st.sidebar.slider("Number of recent tweets", 10, 1000, 100)

#         if handle and api_key and api_secret and access_token and access_secret:
#             auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
#             api = tweepy.API(auth)

#             tweets = []
#             for tweet in tweepy.Cursor(api.user_timeline, screen_name=f"@{handle}", tweet_mode="extended").items(tweet_limit):
#                 tweets.append([tweet.created_at, tweet.full_text])

#             df = pd.DataFrame(tweets, columns=['date', 'text'])
#     except Exception as e:
#         st.error(f"Error loading Twitter data: {e}")

# elif data_source == "Reddit Subreddit":
#     try:
#         import praw
#         st.sidebar.markdown("**Reddit API Credentials**")
#         reddit_client_id = st.sidebar.text_input("Client ID", value="7GwE5Bdr9d6S6zughGa8rA")
#         reddit_client_secret = st.sidebar.text_input("Client Secret", value="Ng-aZ0kN8ZNdvTBRwDU9ntPd9yD-rA")
#         reddit_user_agent = st.sidebar.text_input("User Agent", value="sentiment-analysis-script")

#         subreddit_name = st.sidebar.text_input("Enter subreddit name (without /r/):")
#         post_limit = st.sidebar.slider("Number of posts", 10, 500, 100)

#         if subreddit_name and reddit_client_id and reddit_client_secret and reddit_user_agent:
#             reddit = praw.Reddit(client_id=reddit_client_id,
#                                  client_secret=reddit_client_secret,
#                                  user_agent=reddit_user_agent)

#             posts = []
#             for post in reddit.subreddit(subreddit_name).new(limit=post_limit):
#                 # Get the post text
#                 post_text = post.title + " " + post.selftext
#                 post_data = [datetime.fromtimestamp(post.created_utc), post_text]
#                 posts.append(post_data)

#                 # Fetch comments and analyze them as well
#                 post.comments.replace_more(limit=0)  # Replace "MoreComments" object
#                 for comment in post.comments.list():
#                     comment_data = [datetime.fromtimestamp(comment.created_utc), comment.body]
#                     posts.append(comment_data)

#             df = pd.DataFrame(posts, columns=['date', 'text'])
#     except Exception as e:
#         st.error(f"Error loading Reddit data: {e}")


# In[34]:


# If data is loaded
if not df.empty:
    if 'text' not in df.columns:
        st.error("Data must include a 'text' column.")
    else:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            min_date, max_date = df['date'].min(), df['date'].max()
            selected_range = st.sidebar.date_input("Date range:", [min_date, max_date])

            df = df[(df['date'] >= pd.to_datetime(selected_range[0])) &
                    (df['date'] <= pd.to_datetime(selected_range[1]))]

        # st.sidebar.header("2. Sentiment Settings")
        # model_choice = st.sidebar.selectbox("Choose sentiment model:", ["RoBERTa", "VADER", "Feel-IT"])
        # conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)



        # with st.spinner("Running Sentiment Analysis..."):
        #     MAX_LENGTH = 512
        #     df['text'] = df['text'].astype(str).apply(lambda x: x[:MAX_LENGTH])

        #     if model_choice == "RoBERTa":
        #         roberta_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        #         results = roberta_pipeline(df['text'].tolist())
        #         df['label'] = [r['label'] for r in results]
        #         df['score'] = [r['score'] for r in results]

        #         df = df[df['score'] >= conf_threshold]

        #         label_map = {"LABEL_2": "POSITIVE", "LABEL_1": "NEUTRAL", "LABEL_0": "NEGATIVE"}
        #         score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}

        #         df['sentiment'] = df['label'].map(label_map)
        #         df['sentiment_score'] = df['sentiment'].map(score_map)



        #     elif model_choice == "VADER":
        #         vader = SentimentIntensityAnalyzer()
        #         df['score'] = df['text'].apply(lambda x: vader.polarity_scores(x)['compound'])
        #         df = df[df['score'].abs() >= conf_threshold]

        #         def vader_label(score):
        #             if score >= 0.05:
        #                 return "POSITIVE"
        #             elif score <= -0.05:
        #                 return "NEGATIVE"
        #             else:
        #                 return "NEUTRAL"

        #         df['sentiment'] = df['score'].apply(vader_label)
        #         score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        #         df['sentiment_score'] = df['sentiment'].map(score_map)


        #     elif model_choice == "Feel-IT":
        #         classifier = pipeline("text-classification", model="MilaNLProc/feel-it-italian-sentiment", top_k=1)
        #         results = classifier(df['text'].tolist())

        #         df['label'] = [r[0]['label'].upper() for r in results]  # Get top prediction and make uppercase
        #         df['score'] = [r[0]['score'] for r in results]

        #         df = df[df['score'] >= conf_threshold]  # Filter by confidence threshold

        #         # Only map POSITIVE and NEGATIVE
        #         label_map = {"POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE"}
        #         score_map = {"POSITIVE": 1, "NEGATIVE": -1}

        #         df['sentiment'] = df['label'].map(label_map)
        #         df['sentiment_score'] = df['sentiment'].map(score_map)



        st.subheader("Summary")
        df['sentiment'] = df['sentiment'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        # df['sentiment'] = df['sentiment'].astype(str)
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        # sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].astype(str)  # Just to be extra safe

        col1, col2 = st.columns([1, 1])


        with col1:
            overall_score = df['sentiment_score'].mean()
            st.markdown(
                f"""
                <div style="text-align: center; padding-top: 50px;">
                    <h3>Overall Sentiment Score</h3>
                    <div style="font-size: 48px; font-weight: bold;">{overall_score:.2f}</div>
                    <div style="font-size: 14px; color: gray;">(Ranges from -1 = very negative to +1 = very positive)</div>
                </div>
                """, unsafe_allow_html=True
            )

        with col2:
            fig = px.pie(sentiment_counts,
                         names='Sentiment',
                         values='Count',
                         title='Sentiment Distribution',
                         color='Sentiment',
                         color_discrete_map={'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'},
                         category_orders={"Sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"]})
            fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0)) # Set pie chart height
            st.plotly_chart(fig, use_container_width=True)


        # st.subheader("Summary")
        # sentiment_counts = df['sentiment'].value_counts().reset_index()
        # sentiment_counts.columns = ['Sentiment', 'Count']
        # st.plotly_chart(px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Distribution',
        #       color='Sentiment', color_discrete_map={'POSITIVE': 'green','NEUTRAL': 'gray','NEGATIVE': 'red'}))

        # overall_score = df['sentiment_score'].mean()
        # st.metric("Overall Sentiment Score ℹ️", f"{overall_score:.2f}", help="The average of sentiment scores: +1 for Positive, 0 for Neutral, -1 for Negative. Ranges from -1 (very negative) to +1 (very positive).")

        if 'date' in df.columns:
            time_series = df.groupby([df['date'].dt.date, 'sentiment']).size().reset_index(name='Count')
            st.subheader("Sentiment Over Time")
            st.plotly_chart(px.line(time_series, x='date', y='Count', color='sentiment', title="Sentiment Trends Over Time",
              color_discrete_map={'POSITIVE': 'green','NEUTRAL': 'gray','NEGATIVE': 'red'},
              category_orders={"sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"]}))

        st.subheader("Word Cloud by Sentiment")

        # Ensure necessary NLTK downloads
        nltk.download('punkt')
        nltk.download('wordnet')

        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()



        # Function to clean and preprocess the text
        def clean_text(text):
            text = re.sub(r'[^a-zA-Zàèéìòù\s]', '', text)  # Allow Italian accented letters
            words = text.lower().split()
            words = [word for word in words if word not in italian_stopwords and len(word) > 2]
            return ' '.join(words)

        # Only show POSITIVE and NEGATIVE side by side
        cols = st.columns(2)
        for idx, sentiment in enumerate(['POSITIVE', 'NEGATIVE']):
            # Clean the text before generating word cloud
            texts = " ".join(df[df['sentiment'] == sentiment]['text'])
            cleaned_texts = clean_text(texts)  

            # Apply cleaning function
            if cleaned_texts:
                wordcloud = WordCloud(width=800, height=300, background_color='white').generate(cleaned_texts)

                with cols[idx]:
                    st.markdown(f"#### {sentiment}")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                    # Extract top 10 words from the word cloud
                    word_freq = wordcloud.words_
                    top_words = list(word_freq.items())[:10]

                    if top_words:
                        words, freqs = zip(*top_words)

                        df_bar = pd.DataFrame({'Word': words, 'Frequency': freqs})

                        fig = px.bar(
                            df_bar,
                            x='Frequency',
                            y='Word',
                            orientation='h',
                            title=f"Top Words in {sentiment} Posts",
                            color='Frequency',
                            color_continuous_scale='Blues'
                        )

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                        fig.update_coloraxes(showscale=False) # Keep bars sorted
                        st.plotly_chart(fig, use_container_width=True)





                    # if top_words:
                    #     words, freqs = zip(*top_words)
                    #     fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                    #     import seaborn as sns
                    #     sns.barplot(x=list(freqs), y=list(words), ax=ax_bar, palette="Blues_d")
                    #     ax_bar.set_title(f"Top Words in {sentiment} Posts")
                    #     ax_bar.set_xlabel("Relative Frequency")
                    #     fig_bar.tight_layout()  # ensure consistent padding/margins
                    #     st.pyplot(fig_bar)


        with st.expander("🔍 View Analyzed Posts", expanded=False):
            display_df = df[['date', 'text', 'sentiment', 'score']].copy()
            display_df.columns = ['Date', 'Text', 'Sentiment', 'Confidence']
            st.dataframe(display_df.sort_values(by='Date', ascending=False).reset_index(drop=True), height=400)

        st.subheader("Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")
else:
    st.info("Awaiting data input...")


# In[ ]:





# In[14]:


# xlm_roberta_pipeline = pipeline("sentiment-analysis", model="xlm-roberta-base")
# result = xlm_roberta_pipeline("Oggi sono proprio contento!")
# print(result)


# In[44]:


# classifier = pipeline("text-classification", model="MilaNLProc/feel-it-italian-sentiment", top_k=2)
# prediction = classifier("Oggi sono proprio contento!")
# print(prediction)


# In[57]:


# classifier = pipeline("text-classification", model="MilaNLProc/feel-it-italian-sentiment", top_k=2)
# prediction = classifier("Dice la stessa cosa mio marito sviluppatore. Non ha mai finito la laurea, si è messo a lavorare. In facoltà manco gli facevano aprire un PC, insegnavano linguaggi vecchissimi e alcuni esami di programmazione li dovevano fare su carta e penna, che fa già ridere così.")
# print(prediction)



# In[18]:


# !python -m spacy download it_core_news_sm


# In[ ]:




