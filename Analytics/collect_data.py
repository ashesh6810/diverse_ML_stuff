import tweepy
from tweepy import OAuthHandler
 
consumer_key = 'Ddni33ugXaBysRHM77ywJ7J8e'
consumer_secret = '1hRBK551WIAh4zJ8SFqj9P5NRqNUJmyqZ0fr1zEQWq7byP78DS'
access_token = '845541679226507264-tqKHbCkdfk2GpXTj2KzeXpSNtKciUyU'
access_secret = '6pk30pDwXSaQVyRiYQRTjCCZW4gs6nmq9aE5AbjyUca7X'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

for status in tweepy.Cursor(api.home_timeline).items(10):
 print(status.text)
