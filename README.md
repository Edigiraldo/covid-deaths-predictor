# Files Structure:
**Data**: Directory where all data for the model is stored.
  - **Raw_Data**: Johns Hopkins dataset.
  - **Processed_Data**: Raw Data after being smoothed, cleaned and filter. Dataset is divided by regions.
  - **Training_Data**: Data used for the model to train.

**Process_covid_cases_data.ipynb**: Notebook to preprocess covid cases dataset in Raw_Data.
**Process_covid_deaths_data.ipynb**: Notebook to preprocess death cases dataset in Raw_Data.
**Process_covid_vaccination_data.ipynb**: Notebook to preprocess covid vaccination data from dataset in Raw_Data.

**Filter_training_data.ipynb**: Notebook to divide datasets into countries and filter unwanted samples. Generates data for Processed_Data directory.
**Generate_training_data.ipynb**: Notebook to generate data for training after being filtered. Generates data for Training_Data directory.
**Model_training.ipynb**: Notebook to build and train the model.

# Run the proyect in the following order:
    - Process_covid_cases_data.ipynb
    - Process_covid_deaths_data.ipynb
    - Process_covid_vaccination_data.ipynb
    - Filter_training_data.ipynb
    - Generate_training_data.ipynb
    - Model_training.ipynb

-----------------------------------------
![](RackMultipart20210919-4-1ff7af8_html_237499165a11f2b9.gif)

### **Predicting covid deaths with Neural Networks.**

![](RackMultipart20210919-4-1ff7af8_html_237499165a11f2b9.gif)

By: Robinson Montes, Daniel Pérez , Diego Gómez and Edison Giraldo. Holberton School.

[Github link](https://github.com/Edigi12Hbtn/Covid_Project).

![](https://cdn-images-1.medium.com/max/1600/0*ZofwnkzVe6V57siQ)

Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.

Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age.[1]

The covid-19 virus appeared in December 2019 when it was identified for the first time in Wuhan, China. Since then, the virus started to spread around the world and the World Health Organization (WHO) declared a[Public Health Emergency of International Concern](https://en.wikipedia.org/wiki/Public_Health_Emergency_of_International_Concern) on 30 January 2020, and a pandemic on 11 March 2020 [2]. Since 2019 and to date, more than 219 M cases have been reported and more than 4.55 M people have died worldwide.

The efforts around the globe have been focused on developing vaccines, distributing them, imposing restrictive and preventive measures to limit the spread of the virus and models to predict future spikes of Covid infection have also been implemented to plan the government response. This project aims to develop a model to predict deaths caused by Covid, given the cases of Covid-19 for a country and the vaccination data of its population. We believe these work may be useful in the future when we can predict not only Covid infections, but also the deaths caused by those infections.

### **Step by step process to predict death cases.**

### **Step 1: Review and Tidying the datasets.**

### **Obtain the datasets:**

Datasets are the most relevant point for whatever Machine Learning project. In the case of the current pandemia caused by the Covid-19 virus, the speculation about deaths, positive cases, and vaccination, generates a lot of noise in the real statistics for each country reports. To be sure that the data is closer to reality, the datasets were obtained from Johns Hopkins University[github](https://github.com/CSSEGISandData/COVID-19/) that filters the official reports in World Health Organization -WHO-.

In the following image you can see the vaccination data set in different countries obtained from the John Hopkins data set.

![](https://cdn-images-1.medium.com/max/1600/0*SGKzZWQdsBxFxctr)

Data preprocessing presented us with some challenges. We face problems like lack of data in all data sets, outliers, inconsistent data, noisy series and others.

### **Fix structural difference:**

One of the first problems we had to deal with was that the data was not structured in the same way. Above you can see that the vaccination data was divided into days, countries and even regions. The structure of the dataset for covid cases is shown below. As can be noticed, the general structure of both datasets was totally different, the dates were not even the same. We even found that countries were not the same in the different datasets, and in some cases the registered &#39;Countries&#39; were in fact cruise ships, as was the case of &#39;MS Zaandam&#39; and &#39;Diamond Princess&#39; which had to be manually checked and deleted from datasets.

![](https://cdn-images-1.medium.com/max/1600/1*TSB3t82NVDobQwfenueH7w.png)

### **Handle missing data:**

Another problem we had was that after standardizing the structure of all the data sets, problems such as lack of data appeared on some specific dates. Especially in the vaccination data, we found that data was missing in most cases. What caught our attention was that most of the missing data was on some specific dates. To fill in these values we noticed that a simple forward propagation of values was sufficient. We did not need to use techniques such as linear interpolation because the changes in vaccination in the ranges where data were missing were not so important.

![](https://cdn-images-1.medium.com/max/1600/0*AbHtMu3iophBi-ga)

### **Filter unwanted outliers:**

Some data values were out of range compared to others in some countries. In these cases, these points represented a problem due to the smoothing process that we had to apply to the data to smooth it out. The explanation for these types of points may be that countries sometimes underestimated Covid cases, vaccines or deaths a few weeks before the date of appearance of that outlier, and when countries realize that they underestimated the data , report all past cases not reported in just one day. To fix this, we had to manually go through the dataset and remove these outliers to make the data more suitable to be smoothed in a suitable way to be fed into the neural network. The outliers were replaced by the same value as the previous day.

![](https://cdn-images-1.medium.com/max/1600/0*lfQGJ5qOGP73GNsn) ![](https://cdn-images-1.medium.com/max/1600/0*KDWzfn-b400qN8Q7)

For countries like Ecuador, France, and Italy, their data sets even contained negative data. In these cases, the outliers were replaced by 0. Since all the data were originally in cumulative values, we assume that these negative values were caused by typographical errors when inserting data, for example, reporting cases without including the correct number of zeros.

![](https://cdn-images-1.medium.com/max/1600/0*KRHpOE3vOvUuBrug)

### **Apply a smoothing algorithm**

As shown in the image above in the case of the data for France, some countries showed a periodic pattern with a lower index on weekends and higher values on weekdays due to under reporting of cases on Saturdays and Sundays . Because this noisy data is not as representative of the evolution of Covid cases, we decided to smooth the data with a technique called exponentially weighted average.

![](https://cdn-images-1.medium.com/max/1600/0*t3cbElJg12qC9bZ5)

### **Step 2: Normalize and split the data.**

### **Normalize the data:**

Once we processed the data to make it more fluid and cleaner, we discovered that the data was not yet as appropriate to be fed into a neural network. Typically, daily values for Covid cases in the dataset range from a few units to hundreds of thousands in large countries, values for deaths were generally one-twentieth of the values for Covid cases, whereas data for fully and partially vaccinated people were generally in the range of millions or even hundreds of millions of people.

To make the data more appropriate to be processed for a neural network, we decided to normalize the Covid cases with respect to the maximum value and the vaccination data with respect to the entire population.

![](https://cdn-images-1.medium.com/max/1600/0*1_A9Su738LTlQIHQ)

Regarding the death data, we had to normalize it in a way that was proportional to the vaccination data, and we decided that the best normalization factor we could take would be a naive estimate of the maximum Covid death based on the maximum number of reported Covid cases. The latter assumption stems from the fact that the country with the worst death outcome was assumed to have no more than 10% of deaths compared to reported cases every day.

### **Split data by regions:**

When analyzing the data after pre-processing and normalization, we found that not all countries exhibited the same behavior with respect to Covid cases vs Covid deaths. One of the observations was that countries that faced the pandemic first, had a higher death rate at its first peak than countries where the pandemic arrived later. In the image below can be seen how in Latin American countries, the proportion in Covid deaths and cases is kept in all different peaks (For getting the real proportion between Covid cases and deaths in the graphs down below, you have to divide by 10 Covid deaths):

![](https://cdn-images-1.medium.com/max/1600/0*cZNjKgZmkf1bxAgS)

While European countries which suffered from Covid some months before, had non-proportional Covid deaths with respect to Covid cases in their first peak compared to the following peaks.

![](https://cdn-images-1.medium.com/max/1600/0*FRGYLmY4erHpi-at)

To get better results when predicting Covid deaths and after seeing this marked behavior in different regions of the world, we decided to divide the data into different geographic regions: Asia, Europe, Africa, North and Central America, South America, Oceania. A better distribution of countries can be considered in future improvements.

### **Step 3: Train the Model**

#### **The model architecture**

To train the model to predict Covid deaths based on vaccination rates and Covid cases in countries, we used a relatively simple architecture. Our model consisted of 3 layers of neural networks. The first is an LSTM recurrent neural network (NN) with 32 hidden units, the second layer is a dense neural network with also 32 units, and the last layer is an output layer composed of a single neuron. We also tried some other architectures like GRU, Bidirectional and multilayer LSTMs and also RNNs but they did not show a significant improvement. The NN was tuned for 100 epochs, with the mean\_absolute\_percentage\_error as a metric, Huber loss and Adam optimizer.

![](https://cdn-images-1.medium.com/max/1600/0*ObZqfqUj4dRxbKJE) ![](https://cdn-images-1.medium.com/max/1600/0*4CkvY4eSsGRQYfE0)

### **Results**

#### **Some specific cases to work on**

First, we must keep in mind that due to the nature of the data, there were a limited number of training examples that could be used. Due to the division that was made to improve the model results, it further limited the number of examples for each region. Below you can see the number of training examples used for each set.

![](https://cdn-images-1.medium.com/max/1600/0*3XyAFKUZ8QOT8NJm)

One approach that can be taken in the future to improve the quantity of data, would be not to take small countries to avoid noisy data, would be to split countries with many population into sub regions. For example, instead of taking the US as a country, it can be divided into states.

![](https://cdn-images-1.medium.com/max/1600/0*0X2t309NxUewNHt_) ![](https://cdn-images-1.medium.com/max/1600/0*8V_PYUp3jowAuiCv)

Above are results for Singapore and Laos, where NN was unable to adequately predict deaths. Covid cases that were rescaled by a factor of 1/40 are shown in blue to make them be on the same scale as deaths. The real deaths are shown in red and the predicted deaths given by the model are shown in green, both in real scale. As can be seen, in the first case, the NN could not predict the low number of deaths in Singapore despite the data being used for training, the second case is similar, and all similar cases in the model present a pattern and it is that the deaths in those countries were quite low or almost non-existent. To solve this problem, the NN could also be fed with metrics about the health system in the countries, the population they have, or even metrics such as per capita gross domestic product can be used.

![](https://cdn-images-1.medium.com/max/1600/1*9Qv9Q21SAXyPjl43f22ukw.png)

As shown in the case of Peru and other countries with high death rates, the NN also faced problems finding a better scale for predicting deaths. This type of country shows the lack of input metrics that can guide the NN to find an appropriate scale for predicting deaths from covid. In future enhancements when we are trying to predict future covid deaths, the NN could be fed with some initial information on actual deaths that could help the neural network achieve better results in these cases.

One final challenge to work on is getting better data or working on better ways to filter it. Below is an issue found with some samples, where even though covid cases were quite low when the pandemic started, secondary reports on covid cases do not provide the NN with adequate information to guess the true scale of deaths by covid.

![](https://cdn-images-1.medium.com/max/1600/0*OJwPHPZQBpsi7_Co)

#### **Achievements**

![](https://cdn-images-1.medium.com/max/1600/0*TVGqguqwBUuVcfQT) ![](https://cdn-images-1.medium.com/max/1600/1*ylWPn0K7mmiqINZmSo9veQ.png) ![](https://cdn-images-1.medium.com/max/1600/0*hSAr45ztV33rNr8j) ![](https://cdn-images-1.medium.com/max/1600/1*ezIgREuSKqEyhxGhc8rDlA.png)

First, there is the relative success of the model in predicting deaths on the same scale as the actual death cases in most samples. It is also important to note how the predicted death peaks in each country always shift to the right compared to the peaks related to covid cases. The latter makes sense considering that people usually die a few weeks after getting sick. There is another quite important characteristic of the predicted death curves, and that is that the covid curves always have almost the same shape as the cases, a common characteristic that is found in almost all predictions.

### **Conclusions**

- We were able to overcome problems in the original dataset with missing data, clean up the noise, identify inappropriate samples, and generally pre-process and standardize the data provided.
- We believe the approach of using NN to predict COVID deaths can be successful, but the biggest challenge is finding more appropriate metrics and getting better, cleaner data.
- There are some issues to work on, such as the need to look for metrics that can give the NN a better way to estimate the scale of predicted deaths, find a better way to divide the data into different sets better than just continents, and handle the predictions for countries where COVID deaths did not exist or were low.
- Despite the problems we encountered, we can show future approaches that can be taken to overcome it and we were able to build a model that predicts covid deaths as expected in most cases.

### **References**

[1][https://www.who.int/health-topics/coronavirus#tab=tab\_1](https://www.who.int/health-topics/coronavirus#tab=tab_1)

[2][https://en.wikipedia.org/wiki/COVID-19\_pandemic](https://en.wikipedia.org/wiki/COVID-19_pandemic)
