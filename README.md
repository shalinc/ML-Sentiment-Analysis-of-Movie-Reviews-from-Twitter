# ML-Sentiment-Analysis-of-Movie-Reviews-from-Twitter

### STEPS TO RUN THE CODE:
	1. The Data folder consists of all the Datsets that have been used in the Project.
	2. The twitter data to be generated needs security credentials.
	3. After getting the credentials replace it in config.json file present in Code folder.
	4. To Run the Movie Review Classifiers type on the console,
		python sentiment_analysis.py -h
			-it will show the help to run the file,
			-The code runs with 3 arguments DatabaseName, Algorithm, Cross Validation
			-usage: ReadingTrainingData.py [-h] dataset algo CV
				positional arguments:
				  dataset     Dataset to Classify (rottom batvsuper junglebook zootopia
							  deadpool)
				  algo        Classification Algorithm to be used (all gnb svm maxEnt)
				  CV          Using Cross validation (yes/no)

		example: to run the code for dataset junglebook using SVM and without crossvalidation type:
			python sentiment_analysis.py junglebook svm no
			
		example: to run the code for dataset rottom using all the classifiers and with crossvalidation type:
			python sentiment_analysis.py rottom all yes
