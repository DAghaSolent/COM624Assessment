<h1>Financial Stock Analysis Software Artefact Solution</h1>

This repository stores my software solution that has been built to complete Financial Stock Analysis tasks that was proposed with the specific requirements outlined in the COM624 assessment brief.

**<p>Demonstration off my Software Artefact Solution: https://Youtube.com</p>**

<h2>Running the application locally instructions</h2><hr>

In order to run this application locally follow the required steps

<li> Clone this Repository to your specified location by inputting the command below</li>

~~~markdown
git clone https://github.com/DAghaSolent/COM624Assessment
~~~

<li> Install the required libraries to be able to run this application by inputting the command below </li>

~~~markdown
pip install -r requirements.txt
~~~

This Application utilises an open source Python Library called [Streamlit](https://streamlit.io/) which is a useful tool in building interactive applications for Machine Learning and Data Science. I have utilised here to create an interactive GUI which allows the user to navigate around my application and see data analytics in real time provided by the machine learning models loaded on my streamlit application. When running the application it doesn't launch and display the application through your preferred IDE terminal, it will launch the application through your preferred web browser on localhost:8501.

<li> To launch the application, in your terminal type in the following command below </li>

~~~markdown
streamlit run streamlitGUI.py
~~~

Once that command is inputted automatically your prefered broswer will open and you will then be greeted with the home page off my application.

![Streamlit Application running](https://github.com/DAghaSolent/COM624Assessment/assets/91552769/e41e3215-a897-4a09-a386-e67b16d890bf)

<hr>

This application is split up into two files the ```main.py``` file which houses all the machine learning models and code to do the relevant tasks set out in the assessment brief. The other file is the ```streamlitGUI.py``` file which handles all GUI side of things for my application and the interactions between my Streamlit GUI and the main.py file. 
<hr>
<h2>Screenshots</h2>

![EDA Screenshot](https://github.com/DAghaSolent/COM624Assessment/assets/91552769/e4480c2f-fd99-4377-8ec5-b8973e7343fa)

![FB Prophet](https://github.com/DAghaSolent/COM624Assessment/assets/91552769/6039e490-a4f1-49cb-9acd-c676e62a2911)

![GeneralTask6Screenshot](https://github.com/DAghaSolent/COM624Assessment/assets/91552769/bb2d6f78-60d6-4d1e-a4a2-cfc9ad9b65a7)

