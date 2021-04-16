# CPSC599GroupProject
Before starting, please run

pip3 install kivy
pip3 install --user --upgrade tensorflow

To train each model:

SVM: python3 
CNN: Visit this google colab link: https://colab.research.google.com/drive/1iv7WFsqMX9SASDNnGordLPn26rPWMxbh?usp=sharing 
Transfer Learning Partial: python3 trainTransferPartial.py 
Transfer Learning Full: python3 trainTransferFull.py
MobileNet: python3 trainMobileNet.py


You do not need to train the models first; we already have the trained models saved. You can skip to trying the application if you would like to but you can train if you wish.

classes = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant']

To run the application, go to the project directory "CPSC599GroupProject"

From there, run the following command: python3 GUIComponent.py

You can then select a button to run the model of your choice to test out your drawings. It does take some time so make sure to click the button once and to wait until the canvas appears. If you want to change models, close the current running canvas and then select another model.

** NOTE **
Not all drawings are recongized that well for each class. When we test our application, these classes are best recognized:
SVM: Airplane, Alarm Clock, The Eiffel Tower, Angel
CNN: Airplane, The Eiffel Tower, The Mona Lisa
Transfer Learning Partial, Transfer Learning Full, MobileNet: Airplane, The Mona Lisa, The Eiffel Tower, Ambulance

You can try to draw the other classes, but there is no gurantee it will be predicted too accurately.
