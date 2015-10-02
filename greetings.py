import random
from gtts import gTTS
import os



class Greetings:

    def __init__(self):

        self.test_phrase = [
            "I Am Your Father!",
            "Fear is The Path To The Dark Side",
            "Live Long and Prosper",
            "Big Data Is Watching You",
            "Welcome To The Desert Of The Real",
            "I Am Your Huckleberry",
            "My Name Is Bond. James Bond",
            "You Shall Not Pass!",
            "Houston, We Have a Problem",
            "Resistance Is Futile",
            "May The Horse Be With You"
            ]



    def test(self):
        tts = gTTS(text=self.test_phrase[int(random.uniform(0,len(self.test_phrase)))],
                   lang='en')
        tts.save('./tts.mp3')
        os.system("mpg321 tts.mp3 -quiet")



    def Say(self, text, lang='en'):
        tts = gTTS(text=text,
                   lang=lang)
        tts.save('./tts.mp3')
        os.system("mpg321 tts.mp3 -quiet")



    def GenerateGreeting(self, text, lang='en'):
        _greeting = '{}. {}'.format(
            text,
            self.test_phrase[int(random.uniform(0,len(self.test_phrase)))]
            )
        return _greeting


    def Greet(self, text, lang='en'):
        tts = gTTS(text=self.GenerateGreeting(text),
                   lang=lang)
        tts.save('./tts.mp3')
        os.system("mpg321 tts.mp3 -quiet")

