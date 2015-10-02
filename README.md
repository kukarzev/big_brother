Pearson Hackathon 2015
======================
Pearson office in Centenial, Colorado

Pre-requisites
--------------
    pip install -r requirements.txt
    sudo apt-get update
    sudo apt-get install mpg321


Modules
-------
greetings.py - text-to-speech using Google TTS API
    import greetings
    g = greetings.Greetings()
    g.Say("I am your father!")

capture_training.py - capture training images

webcam_find.py - main app