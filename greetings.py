import random
from gtts import gTTS
import os



class Greetings:

    def __init__(self):

        self.test_phrase = [
            "I am your father!",
            "There is no spoon",
            "Fear is the path to the Dark Side",
            "Run, fools!",
            "Nobody calls me chicken!",
            "Live Long and Prosper",
            "May the horse be with you, always"
            ]

    def test(self):
        tts = gTTS(text=self.test_phrase[int(random.uniform(0,len(self.test_phrase)))],
                   lang='en')
        tts.save('./tts.mp3')
        os.system("mpg321 tts.mp3 -quiet")



    def Say(self, text):
        tts = gTTS(text=text,
                   lang='en')
        tts.save('./tts.mp3')
        os.system("mpg321 tts.mp3 -quiet")
