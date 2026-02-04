# Get started with Machine Learning

Before you get into the AI projects that are the heart of this book, this chapter shows you some fundamental machine learning techniques that will introduce you to modelling and inference. 

Before you run these, you should [create a virtual environment](https://www.raspberrypi.com/news/using-python-with-virtual-environments-the-magpi-148/) and then run the command `pip install -r requirements.txt`.

You should run the first notebook, [Collect CPU Data](<./Collect CPU Data.ipynb>), on your Raspberry Pi. It will collect data on CPU speed, fan speed, CPU usage, CPU temperature, and voltage. Use your Raspberry Pi as normal while it's running, but please try to do something CPU-intensive once or twice, such as playing one of the games from Code the Classics (click the Raspberry Pi icon, choose Preferences, then use the Recommended Software tool to install them).

When you're done, you'll have a timestamped readings file; you can rename it to readings.csv and use it with the second notebook, [Analyse CPU Data](<./Analyse CPU Data.ipynb>). If you want, you can copy the CSV file and notebook over to another computer and run it there, but the Pi 5 should run it no problem. That notebook attempts to create models that predict the CPU temperature and whether the fan is on. You'll find complete detail in the book, AI projects with Raspberry Pi (ISBN 978-1916868427, expected in Summer of 2025).

It's entirely possible that the model will perform poorly with the data you collect. That is one of the hazards of the trade. However, the readings.csv file included in this repository were gathered in real-world conditions, and have not been modified. When we trained the model with them, we observed quite reasonable predictive results.