# SLI-App
Mobile spoken language classifier.

## About this project
This project is made for the "Short Programming Project" course initiated by the University of Groningen. The final product is a mobile app that allows users to speak any sentence, and the app with tell them in which language they spoke. The app is trained only on male voices in Dutch, English, German, French, Spanish, and Italian. 

## Technical details
Under the hood, this app uses a Convolutional Neural Network. The network is relatively small, so it can run quickly on a mobile device. The network is written in the PyTorch C++ API (a.k.a. LibTorch), which allows it to be compiled into a single binary. 
The network is trained on a subset of the Mozilla Common Voice dataset, totalling over 3500 mp3 samples. The samples have a length between about 2 and 4 seconds. The network splits each sample in sections of about 0.25 seconds, and classifies each independently. The average result is the output of the network.

## Statistics
!Disclaimer! the app is still under development, statistics may not be final.
The classifier can classify testing data in mp3 format of theoretically any length. On the 3500+ training samples, we obtain:
* Classification accuracy: 0.899 (89.9%)
* Average Natural Log Loss: 1.09
* Loss Standard Deviation: 0.82
