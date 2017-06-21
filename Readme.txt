After unzipping the flask-app-master folder,
make sure to save flask-app-master folder to D drive



path to folder must be D://flask-app-master

{check:
 zip some times creates double folders inside, please extract flask-app-master containing 
all sub folders

}

D://flask-app-master//
.........core
          .....code
.........GDADATA
	  .....GDA_DATA.csv
.........images(sample images >20mb) just to creat hyperlink in pdf for interactive visualizations
.........static
	  .....css
.........templates
	  .....layouts
		......index.html
	  .....form.html
	  .....image.html
..........app.py
..........reuirements.txt
..........sql_code
..........uber_vk.ipynd

python 3.6.0
find requirements.txt file and pip install -r requirements.txt in python shell

---Analysis_vk.pdf is analysis file

---to run the application open command prompt
	@D:\flask-app-master>python app.py

	@Once following commands appear
 	* Running on http://127.0.0.1:4762/ (Press CTRL+C to quit)
 	* Restarting with stat
 	* Debugger is active!
	 * Debugger pin code: 242-451-845

	if any package not found open uber_vk.py file and make sure all the import packages present 
	are installed

	open chrome web browser and type localhost:4762

	Choose from tabs customer segment, route analysis, trend analysis

---uber_vy.ipynb is jupyter notebook where raw code is present beside application(tool)
	@jupyter notebook helps to understand all the functions in project.
	@to install jupyter notebook pip3 install jupyter
	     @open command prompt
		--D:\flask-app-master>jupyter notebook
		 	copy and paste localhost:      port number running in command prompt
		
