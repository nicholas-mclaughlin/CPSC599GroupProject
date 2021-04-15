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
		layout = GridLayout(rows=4)
		selectAlgoText = Button(text = "Select your machine learning algorithm to test our drawings:")
		transferLearningButton = Button(text ="Transfer Learning and Mobile Net")
		CNNButton = Button(text ="CNN")
		SVMButton = Button(text = "SVM")
		# bind() use to bind the button to function callback
		transferLearningButton.bind(on_press = self.callbackForTL)
		CNNButton.bind(on_press = self.callbackForCNN)
		SVMButton.bind(on_press = self.callbackForSVM)
		layout.add_widget(Label(text='Select your machine learning algorithm to predict what you are drawing:'))
		layout.add_widget(transferLearningButton)
		layout.add_widget(CNNButton)
		layout.add_widget(SVMButton)
		return layout


	# callback function tells when button pressed
	def callbackForTL(self, event):
		os.system('python3 sketch.py')

	def callbackForCNN(self, event):
		os.system('python3 sketch_one_channel.py')

	def callbackForSVM(self, event):
		os.system('python3 sketchSVM.py')
		
		

# creating the object root for ButtonApp() class
root = ButtonApp()

# run function runs the whole program
# i.e run() method which calls the target
# function passed to the constructor.
root.run()
