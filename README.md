# Topic modeling

This project uses Natural Language Processing and Machine Learning methods to achieve multi-class classification problem. 

## 🔑 Method
In this project, I use approximately ten-thousand descriptions of podcasts broadcasting in *Spotify* which are obtained by API. Together with their associated label, the data is used to trained the models to classify the label from text. All the labels are pre-defined by me, currently 10 classes, including `["machine learning", "cooking", "crime", "politics", "kid", "comedy", "sport", "culture", "lifestyle", "business"]`.

Once the models are done, the API will be created and deployed to serve models as a service. Then I create a simple web interface for user to interact with models. After that, the web interface will be deployed to public.

## 📑 Action plan
1. **Preparing data**
    - **Getting data from Spotify API** ✅
      - Ingest data into `.json`  then tabulate into `.csv`
      - Artifact : 
        1. `included.txt` : log for raw data that is in conditions
        2. `excluded.txt` : log for raw data that is not in conditions 
        3. `failed_ep_query.txt` : log for failed API calls
        4. `total_df.csv` : resulting table containing infomation useful for analysis
    - **Cleansing data** ✅
      - Drop duplicated and missing values
      - Mask url as `[UNK]` token
      - Artifact :
        1. `df_notnull_notdup.csv` : cleansed version of `total_df.csv`
    - **Text preprocessing** 🔜
      - Does preprocessing the text can improve the model's performance ?
2. **Building model**
    - Without text preprocessing
        |     Model     |               Method              | Status | Overall accuracy |
        |:-------------:|:---------------------------------:|:------:|:----------------:|
        |    Baseline   |         Sklearn estimators        |    ✅   |        50%       |
        | Deep Learning |   Sequential's Tensorflow model   |    ✅   |        70%       |
        |  Transformer  | Fine tune pre-trained HuggingFace |    ✅   |        80%       |
    - With text preprocessing 🔜
3. **Create backend API** to serve ML as a service
    - **FastAPI**
      - Recieve a sentence and return predicted probabilities of each class 🔛
      - ...
    - **Automated testing**
      - ...
    - **Deploy backend** as Microservice
      - ...
    - ...

4. **Create web frontend** as a user's interface to API
    - **Deploy web interface**
      - ...
    - ...

5. **Orchestraion(K8)**
    - ...
    
## 📍 Acknowledgement
- All the datasets, log files, and models are not included in this repository.

