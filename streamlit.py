import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob
from spellchecker import SpellChecker
from langdetect import detect
import docx2txt
from PIL import Image
import pdfplumber
import PyPDF2
import re
from PyPDF2 import PdfReader
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import os
import string
from nltk.tokenize import word_tokenize
import re
 #Docx resume
import docx2txt
 #Wordcloud
import re
import operator
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
set(stopwords.words('english'))
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#from multiapp import Multiapp

import base64

#st.sidebar.title('Page: Information')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
 

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
st.balloons()

#Punctuation function.
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"',"'"))
    return final

import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

from nltk.corpus import stopwords
stopword_list=nltk.corpus.stopwords.words('english')
STOPWORDS = set(stopwords.words('english'))
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


jd_list1=[]

mid, col1, col2 = st.columns([20,20,20])
with col1:
    st.markdown(f'<h4 style="color:#8B008B;font-size:42px;">{"WELCOME TO AUGUR WORLD"}</h4>', unsafe_allow_html=True)

image= Image.open('HR.jpeg')

mid, col1, col2 = st.columns([20,20,20])
with col1:
    st.image(image, width=500)
#with col2:
#    st.markdown(f'<h4 style="color:#8B008B;font-size:42px;">{"WELCOME TO AUGUR "}</h4>', unsafe_allow_html=True)
    #st.write('Welcome to Augur',unsafe_allow_html=True)
    

#options_upload = ""
#uploadedoptions = st.multiselect(
     #'JOB DESCRIPTION',
#['data','modelling','code','visualization','cloud','analytics','Java','R','database','NLP','BI','IOT','python','AI','predictiveanalytics','programming','API','statistics','sql','clustering','hadoop','chatbot','big data', 'AWS','data mining','machine learning', 'prescriptive analytics', 'diagnostic analytics', 'excel', 'Power Point', 'PowerPoint'])

#st.sidebar.write('Please select choices above or:')

#if st.sidebar.button('Upload your own JD doc'):
 #   options_upload = st.sidebar.file_uploader("Job Descriptions Upload - Word File", type = ['docx'])
  #  if options_upload is not None:
   #     uploadedoptions = docx2txt.process(options_upload)
   # else: 
    #    options_upload = st.sidebar.file_uploader("Job Descriptions Upload - Word File", type = ['pdf'])
     #   if options_upload is not None:
      #      uploadedoptions=PdfReader(options_upload)
       #     for page in reader.pages:
        #        text += page.extract_text() + "\n"
            #with pdfplumber.open(options_upload) as pdf:
             #   page = pdf.pages[0]
             #   uploadedoptions = st.write(page.extract_text())
                


                
#dataset = ""
#df1 = ""

Total=0
#dataset = st.file_uploader("Upload Resume - Word File", type = ['docx'])
#if dataset is not None:
#    df1 = docx2txt.process(dataset)
    
    
#dataset = st.file_uploader("Upload Resume - PDF File", type = ['pdf'])
#if dataset is not None:
#    with pdfplumber.open(dataset) as pdf:
#        text = ""
#        reader = PdfReader(dataset)
#        for page in reader.pages:
#            text += page.extract_text() + "\n"
#            df2=text

            
skills=['data','python','modelling','analytics','tableau','coding','excel','visualization','chatbot','Machine Learning','sql','database','programming','dashboard','cloud','predectiveanalysis','mining','nlp','aws','Power BI','Deep Learning','Power Point','statistics','code','Neural Networks','apache','modeling','Big Data','api','hadoop','Microsoft Office']
import streamlit as st
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Enter the dataset", "Resume Print","Skill Score", "Applicable Candidate for JD","Accept/Reject of Offer"])
                                        
                                        
with tab1:
    st.balloons()
    #st.header("Enter the dataset")
    #col1, col2, col3 = st.columns(3)
    col1, col2, col3 = st.columns([20,20,20])
with col1:
    st.header("Skills or JD")
    
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Please select Skills "}</h4>', unsafe_allow_html=True)
    options_upload = ""
    uploadedoptions = st.multiselect(
     '',
['data','python','modelling','analytics','tableau','coding','excel','visualization','chatbot','Machine Learning','sql','database','programming','dashboard','cloud','predectiveanalysis','mining','nlp','aws','Power BI','Deep Learning','Power Point','statistics','code','Neural Networks','apache','modeling','Big Data','api','hadoop','Microsoft Office'],key=1)
   

    jd1=""

    st.markdown(f'<h4 style="color:#FF0000;font-size:24px;">{" OR "}</h4>', unsafe_allow_html=True)
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Upload JD in docx or pdf "}</h4>', unsafe_allow_html=True)
    #st.header("Upload JD in docx or pdf")
    #st.session_state['key'] = random.randint(0,99999)
    #if st.sidebar.button('Upload your own JD doc'):
    options_upload = st.file_uploader("", type = ['docx'], key="2")
    if options_upload is not None:
        jd1 = docx2txt.process(options_upload)
    else: 
        options_upload = st.file_uploader("Job Descriptions Upload - Word File", type = ['pdf'], key="3")
        if options_upload is not None:
            uploadedoptions=PdfReader(options_upload)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                jd1= text
                
with col2:
    current_salary=0
    expected_salary=0
    offered_salary=0
    experience=0
    st.header("Other Information Required")
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Current CTC of Candidate "}</h4>', unsafe_allow_html=True)
    #st.header("Current CTC of Candidate")
    st.write("Enter as e.g. 15 lacs as 1500000")
    #user_input = st.text_input("",current_salary,key="10")
    current_salary = st.text_input("",current_salary,key="10")
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Expected CTC of Candidate "}</h4>', unsafe_allow_html=True)
    #st.header("Expected CTC of Candidate")
    st.write("Enter as e.g. 15 lacs as 1500000")
    expected_salary = st.text_input("",expected_salary,key="11")
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Offered CTC of Candidate "}</h4>', unsafe_allow_html=True)
    #st.header("Offered CTC of Candidate")
    st.write("Enter as e.g. 15 lacs as 1500000")
    offered_salary = st.text_input("",offered_salary,key="12")
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Experience "}</h4>', unsafe_allow_html=True)
    #st.header("Experience")
    st.write("Should be entered to closest complete value. E.g. Enter 5 years 10 month as 6, 7 months as 1")
    experience = st.text_input("",experience,key="13")
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Is the job location same or not "}</h4>', unsafe_allow_html=True)
    st.write("Enter 1 for Yes else 0 for No ")
    location_option = st.selectbox('',(0,1))
    
    
    
    #user_input = st.text_input("", default_value_goes_here)

with col3:
    
    st.header("Upload Resume in docx or pdf")
    dataset = ""
    ss1 = ""
    ss2= ""
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Upload Resume in docx "}</h4>', unsafe_allow_html=True)
    #st.header("Upload Resume in docx")
    #file = st.file_uploader("Upload Resume", type = ['docx'], key="4")
    file = st.file_uploader('',type = ['docx'], key="4")
    if file is not None:
        ss1 = docx2txt.process(file)

    st.markdown(f'<h4 style="color:#FF0000;font-size:24px;">{" OR "}</h4>', unsafe_allow_html=True)   
    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Upload Resume in pdf "}</h4>', unsafe_allow_html=True)
    #st.header("Upload Resume in pdf")
    #file = st.file_uploader("Upload Resume", type = ['pdf'],key="5")
    file = st.file_uploader('',type = ['pdf'],key="5")
    if file is not None:
        with pdfplumber.open(file) as pdf:
            text = ""
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                ss1 = text

with tab2:
    st.balloons()
    if ss1 != "":
        st.header("Resume Print")
        st.write(ss1)
    else:
        st.header("No Resume Selected to Print")
        


with tab3:
        st.balloons()
        
        if ss1 != "":
            st.header("Resume Score & Weighted Skill Score")
            if uploadedoptions != []:

            #showing the skills selected by HR
                columns =['Skills Selected from skills']
                #st.write(uploadedoptions)
                df=pd.DataFrame(uploadedoptions,columns=['Skills Selected'])
                st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Skills selected "}</h4>', unsafe_allow_html=True)
                st.dataframe(df)
                #need to change df[Skills Selected] to list.
                skill_candidate_list1 = df['Skills Selected'].tolist()

                #cleaning & arranging for resume in ss1
                ss2= str(ss1)
                ss2= ss2.lower()
                ss2=re.sub(r'[^\w\s]', '', ss2)
                ss2= re.sub('[0-9]+', '', ss2)
                #st.write(ss2)
                #found the count of list present in resume.
                #list1 = ['analytics','project management']
                freqs1 = {}
                for y in skill_candidate_list1:
                    if y in ss2:
                        pair = (y)
                        freqs1[pair] = freqs1.get(pair, 0) + 1
                    else:
                        freqs1[y] = 0
                result_series1 = pd.Series(freqs1)
                #st.write(freqs1)
                #df1=pd.DataFrame(freqs1,columns=['Skills Found'])
                #df3=pd.DataFrame([freqs1])
                df3 = pd.DataFrame({'Skills':result_series1.index, 'Skills Found in Resume':result_series1.values})
                #df3['Skills Found in Resume'] = df3.sum(axis=1)
                st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Skills Found in Resume as per Skills selected "}</h4>', unsafe_allow_html=True)
                sum_of_skills_found= df3['Skills Found in Resume'].sum()  
                st.dataframe(df3,width=2000000)
                #st.markdown(f'<h4 style="color:#008B8B;font-size:24px;">{"Total Skills Found are " ,sum_of_skills_found}</h4>', unsafe_allow_html=True)
                #st.write("Total Skills found are in Resume is ",sum_of_skills_found)
                st.markdown(f'<h4 style="color:#008B8B;font-size:24px;">{"Total Skills found in Resume as per HR requirement is "}{sum_of_skills_found}</h4>', unsafe_allow_html=True)


                #find total sum of words present as per skill list
                freqs2= {}
                for z in skill_candidate_list1:
                    if z in ss2:
                        pair = (z)
                        freqs2[pair] = ss2.count(pair)
                    else:
                        freqs2[z] = 0
                result_series = pd.Series(freqs2)

                df4 = pd.DataFrame({'Skills':result_series.index, 'Skills TOTAL Found in Resume':result_series.values})
                #df4= df4.set_index('Skill','Count')
                #st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" TOTAL SCORE of all SKILLS "}</h4>', unsafe_allow_html=True)

                #st.dataframe(df4,width=2000)
                

                Total = df4['Skills TOTAL Found in Resume'].sum()
                df4['Weighted Score']= ((df4['Skills TOTAL Found in Resume']/Total)*100)
                #df[['Y','X']].apply(lambda x: pd.Series.round(x, 3))
                df4['Weighted Score']=df4['Weighted Score'].round(2)
                df4=df4.sort_values(by=['Weighted Score'],ascending=False)
                st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Weighted Skills in Resume as per Skills selected"}</h4>', unsafe_allow_html=True)
                st.dataframe(df4,width=2000000)
                #df.sort_values(by=['col1'])
            
            if uploadedoptions == []:
                if jd1 != "":
                    jd2= str(jd1)
                    jd2= jd2.lower()
                    jd2=re.sub(r'[^\w\s]', '', jd2)
                    jd2= re.sub('[0-9]+', '', jd2)
                    freqs3 = {}
                    for y in skills:
                        if y in jd1:
                            pair1 = (y)
                            freqs3[pair1] = freqs3.get(pair1, 0) + 1

                    result_series2 = pd.Series(freqs3)
                    #st.write(freqs3)
                    df5 = pd.DataFrame({'Skills Found in JD':result_series2.index, 'Count in JD1':result_series2.values})
                    df5 = df5.drop('Count in JD1', axis=1)
                    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Skills found in JD "}</h4>', unsafe_allow_html=True)
                    
                    st.dataframe(df5)
                    jd_list1 = df5['Skills Found in JD'].tolist() 
                    ss2= str(ss1)
                    ss2= ss2.lower()
                    ss2=re.sub(r'[^\w\s]', '', ss2)
                    ss2= re.sub('[0-9]+', '', ss2)

                    freqs4 = {}
                    for u in jd_list1:
                        if u in ss2:
                            pair3 = (u)
                            freqs4[pair3] = ss2.count(pair3)
                        else:
                            freqs4[u] = 0
                    result_series3 = pd.Series(freqs4)
                    #st.write(result_series3)
                    df6 = pd.DataFrame({'Skills in JD':result_series3.index, 'Skills Total Found in Resume':result_series3.values})
                    #st.dataframe(df6)
                    Total = df6['Skills Total Found in Resume'].sum()
                    st.markdown(f'<h4 style="color:#008B8B;font-size:24px;">{Total}{" is the Total Score of Participant on Skills found in Resume "}</h4>', unsafe_allow_html=True)
                    #st.write("Total Score of Participant on Skills found in Resume",Total_jd_Resume)
                    #st.write(Total_jd_Resume)
                    df6['Weighted Score of Resume']= (df6['Skills Total Found in Resume']/Total)*100
                        #sum from JD resume match
                        #Total_jd_Resume = df6['Skills TOTAL Found in Resume'].sum()
                    #df6['Weighted Score']= ((df6['Skills Total Found in Resume']/Total)*100)
                    df6['Weighted Score of Resume']=df6['Weighted Score of Resume'].round(2)
                    df6=df6.sort_values(by=['Weighted Score of Resume'],ascending=False)
                    st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Weighted Skills in Resume from JD"}</h4>', unsafe_allow_html=True)
                    st.dataframe(df6,width=20000)
                  
        else:
            st.header("No Resume for Analysis")


        

    
    
    
with tab4:
    st.snow()
    if ss1 == "":
        st.header("Scores of Candidates Applicable for JD")
    
        df7 = pd.read_csv('skills_resume.csv')
        st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Searching the required skills from below Candidates"}</h4>', unsafe_allow_html=True)
        st.dataframe(df7)
        if uploadedoptions != []:
            df9=pd.DataFrame(uploadedoptions,columns=['Skills Selected'])
            skill_candidate_list2 = df9['Skills Selected'].tolist()
            f= 'Name'
            f1= 'Offer Status'
            skill_candidate_list2.append(f)
            skill_candidate_list2.append(f1)
            #elm_count = len(skill_candidate_list2)
            #st.write(elm_count)
            #skill_candidate_list3=
            #st.write(skill_candidate_list2)
            df8=df7[skill_candidate_list2]
            #st.dataframe(df8)
            #df8.loc[(df8!=0).any(axis=0)]
            #df8 = df8.fillna(0)
            #df8.fillna(0,inplace=True)
            #df8.loc[(df8!=0).any(axis=1)]
            #df8=(~(df8[skill_candidate_list2]!=0).any(axis=0))
            #df8 = df8.loc[df8 != 0]

            #df8.dropna()
            #df8 = df8.loc[:, (df8 != 0).all()]
            #st.dataframe(df8,width=20000)
            st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Candidate found with the Skills Selected"}</h4>', unsafe_allow_html=True)
            df8.to_excel('test1.xlsx', index = False)
            df10=pd.read_excel('test1.xlsx')

            #df9=df9.loc[~(df9==0).all(axis=1)
            df10=df10.loc[(df10!=0).all(axis=1)]
            df10 = df10[df10['Offer Status'] == 'Declined']
            rows = len(df10.axes[0])
            st.markdown(f'<h4 style="color:#008B8B;font-size:24px;">{rows}{" Candidates Found "}</h4>', unsafe_allow_html=True)

            #st.text("Candidate Found are ",rows)
            #df10 = df10[df10['Offer Status'] == 'Declined']

            st.dataframe(df10,width=2000000)

        if uploadedoptions == []:
            f= 'Name'
            col_count=""
            #st.write(jd_list1)
            #st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Skills found in JD "}</h4>', unsafe_allow_html=True)
            #df5 = pd.DataFrame({'Skills Found in JD':result_series2.index, 'Count in JD1':result_series2.values})
            #df5 = df5.drop('Count in JD1', axis=1)
            #st.dataframe(df5)

            jd_list2 = jd_list1

            jd_list2.append(f)
            #st.write(jd_list2)
            df8=df7[jd_list2]
            list_len=len(jd_list2)
            #st.write(list_len)
            #st.dataframe(df11)

            #st.write(col_count)
            #st.markdown(f'<h4 style="color:#4682B4;font-size:28px;">{" Candidate found with the JD Skills"}</h4>', unsafe_allow_html=True)
            df8.to_excel('test1.xlsx', index = False)
            df10=pd.read_excel('test1.xlsx')

            #df9=df9.loc[~(df9==0).all(axis=1)
            df10=df10.loc[(df10!=0).all(axis=1)]
            #df10 = df10[df10['Offer Status'] == 'Declined']
            col_count = df10.shape[1]
            if list_len == col_count:
                rows = len(df10.axes[0])
                st.markdown(f'<h4 style="color:#008B8B;font-size:24px;">{rows}{" Candidates Found with all JD Skills "}</h4>', unsafe_allow_html=True)

            #st.text("Candidate Found are ",rows)

                st.dataframe(df10,width=2000000)
            else:
                st.markdown(f'<h4 style="color:#008B8B;font-size:24px;">{0}{" Candidates Found with all JD Skills "}</h4>', unsafe_allow_html=True)
       
    else:
        st.header("Please refer Skill Score, Resume Print & Accept/Reject Offer Page")
            

        
        

    
    
    
    #df8.dropna()
    
    #st.write(ss1)
with tab5:
    st.snow()
    st.header("Accept/Reject of Offer")
    if ss1 != "":
        df_in = pd.read_csv('skills_resume.csv')
        df = df_in.iloc[:,[4,6,7,8,10,11,12,46]]
        df['Loc_Bin'] = np.where(df['Current Location'] == df['Offered Location'], 1, 0)
        df['Offer_Bin'] = np.where(df['Offer Status'] == 'Accepted', 1, 0)
        df = df.drop(['Current Location', 'Offered Location', 'Offer Status'], axis=1)
        cols = ['Exp (Yrs)','Current Salary', 'Expected Salary', 'Offered Salary', 'Feature_Total', 'Loc_Bin']
        X = df[cols]
        y = df['Offer_Bin']

        from sklearn import datasets
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        logreg = LogisticRegression()

        #rfe = RFE(logreg, 6)
        #rfe = rfe.fit(X, y.values.ravel())

        from sklearn import metrics

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)


        # TO GET IT FROM OTHER TABS
        new_array = np.array(([experience,current_salary,expected_salary,offered_salary,Total,location_option]))

        X_new = pd.DataFrame([new_array], columns= ['Exp (Yrs)','Current Salary', 'Expected Salary', 'Offered Salary', 'Total Skills', 'Loc_Bin'])
        #st.dataframe(X_new)
        y_pred_new = logreg.predict(X_new)
        X_new['Predicted_Offer'] = np.where(y_pred_new == 1, 'Accept', 'Decline')
        st.dataframe(X_new)
        #X_new.to_excel('test11.xlsx', index = False)
        #df100=pd.read_excel('test11.xlsx')
        if y_pred_new == 1:
            st.markdown(f'<h4 style="color:#C71585;font-size:36px;">{" The candidate shortlisted will ACCEPT the offer "}</h4>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h4 style="color:#C71585;font-size:36px;">{" The candidate shortlisted will DECLINE the offer "}</h4>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h4 style="color:#C71585;font-size:36px;">{" No Candidate to Predict "}</h4>', unsafe_allow_html=True)

            
    