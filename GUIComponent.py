# import kivy module
import kivy
import os
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout

# class in which we are creating the button
class ButtonApp(App):
	
	def build(self):
		# use a (r, g, b, a) tuple
		self.title = "CPSC599 Predict Drawings"
		layout = GridLayout(rows=6)
		selectAlgoText = Button(text = "Select your machine learning algorithm to test our drawings:")
		mobileNetButton = Button(text ="Mobile Net")
		transferLearningPartial = Button(text ="Transfer Learning Partial")
		transferLearningFull = Button(text ="Transfer Learning Full")
		CNNButton = Button(text ="CNN")
		SVMButton = Button(text = "SVM")
		# bind() use to bind the button to function callback
		mobileNetButton.bind(on_press = self.callbackForMobileNet)
		transferLearningPartial.bind(on_press = self.callbackForTLPartial)
		transferLearningFull.bind(on_press = self.callbackForTLFull)
		CNNButton.bind(on_press = self.callbackForCNN)
		SVMButton.bind(on_press = self.callbackForSVM)
		layout.add_widget(Label(text='Select your machine learning algorithm to predict what you are drawing:'))
		layout.add_widget(mobileNetButton)
		layout.add_widget(transferLearningPartial)
		layout.add_widget(transferLearningFull)
		layout.add_widget(CNNButton)
		layout.add_widget(SVMButton)
		return layout


	# callback function tells when button pressed
	def callbackForMobileNet(self, event):
		os.system('python3 sketchMobileNet.py')

	def callbackForTLPartial(self, event):
		os.system('python3 sketchTransferPartial.py')

	def callbackForTLFull(self, event):
		os.system('python3 sketchTransferFull.py')

	def callbackForCNN(self, event):
		os.system('python3 sketchCNN.py')

	def callbackForSVM(self, event):
		os.system('python3 sketchSVM.py')
		
		

# creating the object root for ButtonApp() class
root = ButtonApp()

# run function runs the whole program
# i.e run() method which calls the target
# function passed to the constructor.
root.run()
