
# coding: utf-8

# In[68]:


'''
Introduction:

Authors: Sai Nikhitha Madduri and Merin Joy
Date: 8 December 2018
Description: This is a question answering system in Python which can answer questions starting with Where, Who, What, When.
It can answer questions from any domain and provide complete sentences as answers specific to the question if the
user question matches the pre-defined question patterns. If the system finds the output it will give the results
or else it will output sorry, I dont know the answer as the result.

Features Used: Partial Search, Synonym search, POS tagging, sentence splittling.

Results:

Where is Mount Everest located?
Answer: Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas.
Who is A. R. Rahman?
Answer: A. R. Rahman is an Indian music director, composer, singer-songwriter, and music producer.
What is Ecosystem?
Answer: Ecosystem is a community made up of living organisms and nonliving components such as air, water, and mineral soil.
When was Indira Gandhi born?
Answer: Indira Gandhi was born on 19 November 1917
exit
Thank you! Goodbye.


Algorithm:

Step1: Define different types of question patterns using regular expressions.
Step2: Tokenize into words and perform POS tagging to extract the subject for questions starting with Where, who, what
       and spilt the question into entities for questions starting with when.
Step3: Search in Wikipedia with extracted subject.
Step4: Search through the sentences for relevant answers using object and regular expressions.
Step4: Return the most relevant answer to the user as result.

Instructions to run:

1. Import nltk.data module and load nltk.data.load('tokenizers/punkt/english.pickle')
2. Import all the required packages such as wikipedia, re, string, wordnet, sys, nltk.data
3. Run qa-system.py python file along with the log file name in the command prompt as follows,
   $ python qa-system.py mylogfile.txt
4. The question answering system will start, give your questions as input and the system will output the results.

Sample Output:

This is a QA system by Sai Nikhitha Madduri & Merin Joy. It will try to answer questions that start with Who, What, When or Where. Enter exit to leave the program
Where is Mount Everest located?
Answer: Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas.
Who is A. R. Rahman?
Answer: A. R. Rahman is an Indian music director, composer, singer-songwriter, and music producer.
What is Ecosystem?
Answer: Ecosystem is a community made up of living organisms and nonliving components such as air, water, and mineral soil.
When was Indira Gandhi born?
Answer: Indira Gandhi was born on 19 November 1917
exit
Thank you! Goodbye.


'''


# In[69]:


#import all the required packages
import wikipedia
import re
import nltk.data
import string
from nltk.corpus import wordnet
import sys
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[70]:


#Initialize all the wh words, helping verbs and helpers
wh_ques = ["who", "where", "what", "when"]
helping_verbs = ['is', 'are', 'has', 'have', 'has been', 'have been', 'has had', "do", 
                'was', 'were', 'had', 'had been', 'had had', "did",
                'will', 'shall', 'can', 'will have', 'shall have', 'can have',
                'would', 'should', 'could', 'would have', 'should have', 'could have', 'would have been', 'should have been', 'could have been']


# In[71]:


#Take the logfile input from command line
logfile = open(sys.argv[1],"a+")


# In[74]:


#Define function get answer
def get_ans(user_ques):
    
    # Append user question to the log file
    logfile.write('%s\n\n' %user_ques)
    #create a regular expression pattern to seggregate the user question
    re_1 = re.compile(r'('+"|".join(wh_ques)+') ('+"|".join(helping_verbs)+') (.*)\?', re.IGNORECASE)
    #check if regular expression matches the user question
    m = re_1.match(user_ques)
    
    if(m is None):
        return
    #If match is found assign the the words as wh, hv and object
    wh = m.group(1)
    hv = m.group(2)
    obj = m.group(3)
    
    #If the question starts with is Where, Who and What  
    if(wh.lower() != "when"):
        # tokenize the obj to extract noun from it
        tagged_data = nltk.pos_tag(nltk.word_tokenize(obj))
        #Append POS tagged string to the log file
        logfile.write(str(tagged_data))
        # Extract the noun and verb present in the sentence
        noun = " ".join([word_tag[0] for word_tag in tagged_data if word_tag[1]=="NNP"])
        verbs = [word_tag[0] for word_tag in tagged_data if word_tag[1]=="VBD"]
        if(len(verbs) == 1):
            verb = verbs[0]
            #Checking for synonyms of the verb present in the question from wordnet
            syns = [l.name() for syn in wordnet.synsets(verb) for l in syn.lemmas()]
        else:
            verb = ""
        #If verb is present in the question    
        if(verb != ""):
            #Search for the noun in wikipedia
            search_res = wikipedia.search(noun)
            #Append wikipedia search results to logfile
            logfile.write(str('%s\n\n' %search_res))
            
        else:
            #search for object in wikipedia
            search_res = wikipedia.search(obj)
            #Append wikipedia search results to logfile
            logfile.write(str('%s\n\n' %search_res))
        #Store the results in a array called findings 
        findings = []

        for res in search_res:
            #Extract the content present in the first link available on wikipedia
            content = wikipedia.page(res).content
            #Sentence tokenize the data
            sentences = tokenizer.tokenize(content)
            
            if(verb != ""):
                for sent in sentences:
                    # search for the verb present in the question and its synonyms among the tokenized sentences
                    if(re.search('('+verb+'|'+"|".join(syns)+')', sent, re.IGNORECASE) is not None):
                        #search for noun present in the question among the results obtained from above search
                        if(re.search(noun, sent, re.IGNORECASE) is not None):
                            findings.append(sent)
                        else:
                            findings.append(sent)
                            #Append word search results to logfile
                            logfile.write(str('%s\n\n' %findings))
                #If there are no findings display the no answer found sentence
                if(len(findings) == 0):
                    return "I am sorry, I do not know the answer."
                #Print the answer in the format of noun followed by helping verb, followed by our result from find
                else:
                    #Store the first available result in find variable
                    find = findings[0]
                    #Append final sentence extracted to logfile
                    logfile.write(str('%s\n\n' %find))
                    #Print the result
                    return noun+" "+hv+" "+ find[find.index(verb):]
            else:
                for sent in sentences:
                    #search for noun among tokenized sentences
                    if(re.search(noun, sent, re.IGNORECASE) is not None):
                        #search for helping verb among results obtained from above
                        if(re.search(" "+hv+" ", sent, re.IGNORECASE) is not None):
                            findings.append(sent)
                            #Append word search results to logfile
                            logfile.write(str('%s\n\n' %findings))

                if(len(findings) == 0):
                    return "I am sorry, I do not know the answer."
                #store results in findings
                find = findings[0]
                #Append final sentence extracted to logfile
                logfile.write(str('%s\n\n' %find))
                #Print the result
                return obj+" "+ find[find.index(hv+" "):]
    else:
        #When case
        #Split the Object into name and y
        arr = obj.split()
        name = " ".join(arr[:-1])
        y = arr[-1]
        #Search for synonyms of y
        syns = [l.name() for syn in wordnet.synsets(y) for l in syn.lemmas()]
        
        #Search for name in wikipedia
        search_res = wikipedia.search(name)
        #Append wikipedia search results to logfile
        logfile.write(str('%s\n' %search_res))
        findings = []
        
        for res in search_res:
            #Extract the results from first link available on wikipedia
            content = wikipedia.page(res).content
            #Setence tokenize the data
            sentences = tokenizer.tokenize(content)
            for sent in sentences:
                #search for name
                if(re.search(name, sent, re.IGNORECASE) is not None):
                    #search for y and its synonms 
                    if(re.search('('+y+'|'+"|".join(syns)+')', sent, re.IGNORECASE) is not None):
                        if(re.search(" on ", sent, re.IGNORECASE) is not None):
                            findings.append(sent)
                            #Append word search results to logfile
                            logfile.write(str('%s\n\n' %findings))
        if(len(findings) == 0):
            return "I am sorry, I do not know the answer."
        find = findings[0]
        #Append final sentence extracted to logfile
        logfile.write(str('%s\n\n' %find))
        #Search for the date in the string
        sobj = re.search(r'\d{4}', find, re.IGNORECASE)
        year = sobj.group()
        #Print the results
        return name+" "+hv+" "+y+" "+find[find.index("on"):find.index(year)+4]


# In[ ]:


print("This is a QA system by Sai Nikhitha Madduri & Merin Joy. It will try to answer questions that start with Who, What, When or Where. Enter exit to leave the program")
while True:
    
    #Take the user input as user question
    user_ques = input()
    #If user types exit print good bye!
    if(user_ques == 'exit'):
        print("Thank you! Goodbye.")
        #Append output to log file
        logfile.write('%s\n\n' %"Thank you! Goodbye.")
        break
    else:
        # get_answer function is called
        try:
            ans = get_ans(user_ques)
            print(ans)
            #Append output to log file
            logfile.write('%s\n\n' %ans)
            
        #Exception case
        except Exception as e:
            print(str(e))
            print(" I am sorry, I do not know the answer.")
            #Append output to log file
            logfile.write('%s\n\n' %"I am sorry, I do not know the answer.")

