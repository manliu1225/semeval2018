
import jnius_config
import os
_basedir = os.path.dirname(os.path.abspath(__file__))
_parentdir = os.path.dirname(_basedir)
_resources_dir = os.path.join(_basedir, 'resources')

jnius_config.add_options('-Xmx512m', '-XX:ParallelGCThreads=2')
# We set both CLASSPATH environment variable and jnius' internal setting because jnius is kinda cranky.
jnius_config.set_classpath('/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/tweets/resources/ark-tweet-nlp-0.3.2.jar')
print(jnius_config.get_classpath())
# print(*(os.path.join(_resources_dir, jar) for jar in os.listdir(_resources_dir) if jar.endswith('.jar')))
print(os.environ.get('CLASSPATH'))
# os.environ['CLASSPATH'] = u':'.join(jnius_config.get_classpath()) + ((':' + os.environ.get('CLASSPATH')) if os.environ.get('CLASSPATH') else '')
# os.environ['CLASSPATH'] = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/tweets/resources/ark-tweet-nlp-0.3.2.jar'
os.environ['CLASSPATH'] = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/tweets/resources/ark-tweet-nlp-0.3.2.jar'
print(os.environ['CLASSPATH'])

from jnius import autoclass
model_filename = os.path.join(_resources_dir, 'ark_tweet_nlp-20120919.model')  # do not change. problem with pickling and unpickling across filesystems?

Model = autoclass('cmu.arktweetnlp.impl.Model')
print(Model)
print(dir(Model))
print(model_filename)
_model = Model.loadModelFromText(model_filename)