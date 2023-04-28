#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[8]:


pip install deepface


# In[9]:


from deepface import DeepFace


# In[2]:


pip install opencv-python


# In[3]:


img=cv2.imread("happyboy.jpg")


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.imshow(img)


# In[6]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[10]:


predictions=DeepFace.analyze(img)


# In[11]:


predictions


# In[12]:


type(predictions)


# In[13]:


my_list = [{'emotion': {'angry': 3.326193012968999e-12,                        'disgust': 2.113421381514871e-24,                        'fear': 4.64791542113905e-13,                        'happy': 99.99998807907104,                        'sad': 3.6442007744774685e-08,                        'surprise': 1.9929935035634117e-06,                        'neutral': 1.1708446834290953e-05},            'dominant_emotion': 'happy',            'region': {'x': 47, 'y': 16, 'w': 69, 'h': 69},            'age': 21,            'gender': {'Woman': 0.5921975709497929, 'Man': 99.40780401229858},            'dominant_gender': 'Man',            'race': {'asian': 94.45415735244751,                     'indian': 0.44175428338348866,                     'black': 0.005583375605056062,                     'white': 0.9929296560585499,                     'middle eastern': 0.016280572162941098,                     'latino hispanic': 4.0892936289310455},            'dominant_race': 'asian'}]

dominant_emotion = my_list[0]['dominant_emotion']
print(dominant_emotion)  # Output: 'happy'


# In[14]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcasade_frontalface_default.xml')


# In[16]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
             predictions[0]['dominant_emotion'],
             (0,50),
             font,1,
             (0,0,255),
             2,
             cv2.LINE_AA)


# In[17]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[18]:


img=cv2.imread("sad_women.jpg")


# In[19]:


plt.imshow(img)


# In[20]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[21]:


predictions=DeepFace.analyze(img)


# In[22]:


predictions


# In[23]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
             predictions[0]['dominant_emotion'],
             (0,50),
             font,1,
             (0,0,255),
             2,
             cv2.LINE_AA)


# In[24]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[42]:


img=cv2.imread("surprise_man.jpg")


# In[43]:


plt.imshow(img)


# In[44]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[45]:


predictions=DeepFace.analyze(img)


# In[46]:


predictions


# In[47]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
             predictions[0]['dominant_emotion'],
             (0,50),
             font,1,
             (0,0,255),
             2,
             cv2.LINE_AA)


# In[48]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[57]:


import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

actions = ['emotion']  # Define the actions parameter

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions=actions)  # Pass the actions parameter as a keyword argument

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                result[0]['dominant_emotion'],  # Access the dominant emotion from the first element of the list
                (0, 50),
                font, 1,
                (0, 0, 255),
                2,
                cv2.LINE_AA)
    cv2.imshow('Demo video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




