import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 170)   # speed
engine.setProperty("volume", 1.0) # 0.0 to 1.0
engine.say("Hello, this is a text to speech test on Windows.")
engine.runAndWait()