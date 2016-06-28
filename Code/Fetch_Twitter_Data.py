import json
from twitter import Twitter, OAuth


#Configuration File
data = open("config.json")
config = json.load(data)

#Configuration Setup
oauth = OAuth(
    config["access_token"],
    config["access_token_secret"],
    config["consumer_key"],
    config["consumer_secret"]
          )


# Configure the OAuth using the Credentials provided
twitter = Twitter(auth=oauth)

# fetch the Tweets and query accordingly, filtered using links
try:
    iterator = twitter.search.tweets(q='#Junglebook -filter:links',lang='en', count=5000, since="2016-04-12", until="2016-04-13")
except:
    print("ERROR",iterator)


#list which has DICTIONARY for the Tweet JSON
tweets_q = iterator['statuses']

# list of the users who have already tweeted, so as to fetch tweets from different user everytime
users_tweeted = []

# Tweet Count
i = 1

#For every tweet that is fetched, get only relevant tweets
for tweet in tweets_q:
        if (tweet['user']['followers_count'] > 10 and tweet['user']['screen_name'] not in users_tweeted and not tweet['text'].startswith("RT")):
            print(i,' '.join(tweet['text'].split("\n")).encode(encoding='utf-8'))
            users_tweeted.append(tweet['user']['screen_name'])
            i+=1