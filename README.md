# Voice-Activity-Detection
The main goal of this project was to test the classic, ML and DL algorithms for voice activity detection. 
Whole project is based on the record of aerial radiocommication, received via the ICOM IC-A120E aerial radiostation.
The data was recored with sample rate equal to 40 kHz and 14 bit depth. The signal was analyzed by me in Audacity and classified.
Every algorithm operates on frames of signal 20 ms duration each, because 20 ms is the amount of time, where the signal could be considered as static.

The classic VAD algorithms were created based on Frame class, implemented in frame.py file. The ML algorithms were created and testes as well. Each of ML algorithm had the training parameters set based on previous research and GridSearch method.

The RNN neural netork based on LSTM cells was also implemented and trained on the training data set. The depth and size of layers were objects of my research. I tested 36 architectures. The results are avaiable in `lstm_res` folder.


