# Voice-Activity-Detection
The main goal of this project was to test the classic, ML and DL algorithms for voice activity detection. 
Whole project is based on the record of aerial radiocommication, received via the ICOM IC-A120E aerial radiostation.
The data was recored with sample rate equal to 40 kHz and 14 bit depth. The signal was analyzed by me in Audacity and classified.
Every algorithm operates on frames of signal 20 ms duration each, because 20 ms is the amount of time, where the signal could be considered as static.

The classic VAD algorithms were created. <zrc.py> is a script to classify the frames based on Zero Crossing Rate 
