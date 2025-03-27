import feedparser
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
RSS_URL = os.getenv("RSS_FEED_URL")

def read_rss_feed(url):
    feed = feedparser.parse(url)
    
    print(f"Feed Title: {feed.feed.title}\n")
    
    entries_data = []
    
    for entry in feed.entries:
        print(f"Title: {entry.title}")
        print(f"Link: {entry.link}")
        print(f"Published: {entry.published}\n")
        
        entry_dict = {
            "Title": entry.get("title"),
            "Link": entry.get("link"),
            "Published": entry.get("published"),
            "Summary": entry.get("summary")
        }
        entries_data.append(entry_dict)
    
    df = pd.DataFrame(entries_data)
    print("DataFrame of feed entries:\n")
    print(df)

if __name__ == "__main__":
    read_rss_feed(RSS_URL)