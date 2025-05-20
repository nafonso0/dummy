#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import praw
from datetime import datetime
import io


# In[6]:


# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("📊 Sentiment Analysis Tool")


# In[7]:


# Sidebar: Data Input
st.sidebar.header("1. Data Input")
data_source = st.sidebar.radio("Select input source:", ["Upload CSV", "Use Twitter Handle", "Reddit Subreddit"])


# In[8]:


# Load data
df = pd.DataFrame()

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'text' and optional 'date' column:", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

elif data_source == "Use Twitter Handle":
    import tweepy
    st.sidebar.markdown("**Twitter API Credentials**")
    api_key = st.sidebar.text_input("API Key")
    api_secret = st.sidebar.text_input("API Secret", type="password")
    access_token = st.sidebar.text_input("Access Token")
    access_secret = st.sidebar.text_input("Access Token Secret", type="password")

    handle = st.sidebar.text_input("Enter Twitter handle (without @):")
    tweet_limit = st.sidebar.slider("Number of recent tweets", 10, 1000, 100)

    if handle and api_key and api_secret and access_token and access_secret:
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
        api = tweepy.API(auth)

        tweets = []
        for tweet in tweepy.Cursor(api.user_timeline, screen_name=f"@{handle}", tweet_mode="extended").items(tweet_limit):
            tweets.append([tweet.created_at, tweet.full_text])

        df = pd.DataFrame(tweets, columns=['date', 'text'])

elif data_source == "Reddit Subreddit":
    st.sidebar.markdown("**Reddit API Credentials**")
    reddit_client_id = st.sidebar.text_input("Client ID")
    reddit_client_secret = st.sidebar.text_input("Client Secret", type="password")
    reddit_user_agent = st.sidebar.text_input("User Agent", value="sentiment-analysis-script")

    subreddit_name = st.sidebar.text_input("Enter subreddit name (without /r/):")
    post_limit = st.sidebar.slider("Number of posts", 10, 500, 100)

    if subreddit_name and reddit_client_id and reddit_client_secret and reddit_user_agent:
        reddit = praw.Reddit(client_id=reddit_client_id,
                             client_secret=reddit_client_secret,
                             user_agent=reddit_user_agent)

        posts = []
        for post in reddit.subreddit(subreddit_name).new(limit=post_limit):
            posts.append([datetime.fromtimestamp(post.created_utc), post.title + " " + post.selftext])

        df = pd.DataFrame(posts, columns=['date', 'text'])


# In[9]:


# If data is loaded
if not df.empty:
    st.subheader("Sample Data")
    st.dataframe(df.head())

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

        st.sidebar.header("2. Sentiment Settings")
        conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

        st.subheader("Running Sentiment Analysis...")
        MAX_LENGTH = 512
        df['text'] = df['text'].apply(lambda x: x[:MAX_LENGTH])
        results = sentiment_pipeline(df['text'].tolist())

        df['label'] = [r['label'] for r in results]
        df['score'] = [r['score'] for r in results]

        df = df[df['score'] >= conf_threshold]

        label_map = {"LABEL_2": "POSITIVE", "LABEL_1": "NEUTRAL", "LABEL_0": "NEGATIVE"}
        score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}

        df['sentiment'] = df['label'].map(label_map)
        df['sentiment_score'] = df['sentiment'].map(score_map)

        st.subheader("Summary")
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        st.plotly_chart(px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Distribution'))

        overall_score = df['sentiment_score'].mean()
        st.metric("Overall Sentiment Score", f"{overall_score:.2f}")

        if 'date' in df.columns:
            time_series = df.groupby([df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
            st.subheader("Sentiment Over Time")
            st.plotly_chart(px.line(time_series, title="Sentiment Trends Over Time"))

        st.subheader("Word Cloud by Sentiment")
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            texts = " ".join(df[df['sentiment'] == sentiment]['text'])
            if texts:
                wordcloud = WordCloud(width=800, height=300, background_color='white').generate(texts)
                st.markdown(f"#### {sentiment}")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

        st.subheader("Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")
else:
    st.info("Awaiting data input...")

