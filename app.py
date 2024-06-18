# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:15:09 2024

@author: damindu pahasara
"""
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

pt_model= load_model("16_pt.h5")



st.markdown("""
        <h1 style='text-align: center;'>
            😎  Deep Learning-Powered 16 Personalities Test 🤖
        </h1>
    """, unsafe_allow_html=True)

st.write('Welcome to AI-powered personality prediction app!Explore your personality using our AI web app. Answer questions to reveal which of the 16 personality types you align with. It’s easy, private, and helps you understand yourself better.')



# Define the questions and their corresponding encoded scale
questions = [
    "You regularly make new friends.",
    "You spend a lot of your free time exploring various random topics that pique your interest.",
    "Seeing other people cry can easily make you feel like you want to cry too.",
    "You often make a backup plan for a backup plan.",
    "You usually stay calm, even under a lot of pressure.",
    "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.",
    "You prefer to completely finish one project before starting another.",
    "You are very sentimental.",
    "You like to use organizing tools like schedules and lists.",
    "Even a small mistake can cause you to doubt your overall abilities and knowledge.",
    "You feel comfortable just walking up to someone you find interesting and striking up a conversation.",
    "You are not too interested in discussing various interpretations and analyses of creative works.",
    "You are more inclined to follow your head than your heart.",
    "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.",
    "You rarely worry about whether you make a good impression on people you meet.",
    "You enjoy participating in group activities.",
    "You like books and movies that make you come up with your own interpretation of the ending.",
    "Your happiness comes more from helping others accomplish things than your own accomplishments.",
    "You are interested in so many things that you find it difficult to choose what to try next.",
    "You are prone to worrying that things will take a turn for the worse.",
    "You avoid leadership roles in group settings.",
    "You are definitely not an artistic type of person.",
    "You think the world would be a better place if people relied more on rationality and less on their feelings.",
    "You prefer to do your chores before allowing yourself to relax.",
    "You enjoy watching people argue.",
    "You tend to avoid drawing attention to yourself.",
    "Your mood can change very quickly.",
    "You lose patience with people who are not as efficient as you.",
    "You often end up doing things at the last possible moment.",
    "You have always been fascinated by the question of what, if anything, happens after death.",
    "You usually prefer to be around others rather than on your own.",
    "You become bored or lose interest when the discussion gets highly theoretical.",
    "You find it easy to empathize with a person whose experiences are very different from yours.",
    "You usually postpone finalizing decisions for as long as possible.",
    "You rarely second-guess the choices that you have made.",
    "After a long and exhausting week, a lively social event is just what you need.",
    "You enjoy going to art museums.",
    "You often have a hard time understanding other people’s feelings.",
    "You like to have a to-do list for each day.",
    "You rarely feel insecure.",
    "You avoid making phone calls.",
    "You often spend a lot of time trying to understand views that are very different from your own.",
    "In your social circle, you are often the one who contacts your friends and initiates activities.",
    "If your plans are interrupted, your top priority is to get back on track as soon as possible.",
    "You are still bothered by mistakes that you made a long time ago.",
    "You rarely contemplate the reasons for human existence or the meaning of life.",
    "Your emotions control you more than you control them.",
    "You take great care not to make people look bad, even when it is completely their fault.",
    "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.",
    "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.",
    "You would love a job that requires you to work alone most of the time.",
    "You believe that pondering abstract philosophical questions is a waste of time.",
    "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.",
    "You know at first glance how someone is feeling.",
    "You often feel overwhelmed.",
    "You complete things methodically without skipping over any steps.",
    "You are very intrigued by things labeled as controversial.",
    "You would pass along a good opportunity if you thought someone else needed it more.",
    "You struggle with deadlines.",
    "You feel confident that things will work out for you."
]

# Dictionary to store responses
responses = {}

# Display the questions with radio buttons for responses in a horizontal layout
#st.title("16 Personality Test Questionnaire")

#st.title("16 Personality Test Questionnaire")

for i, question in enumerate(questions):
    st.subheader(f"Q{i+1}. {question}")
    option = st.radio("", ['Fully disagree', 'Partially disagree', 'Slightly disagree', 'neutral', 'Slightly Agree', 'Partially Agree', 'Fully Agree'], index=3, key=i)
    
    # Map options to numerical values
    if option == 'Fully disagree':
        numeric_option = -3
    elif option == 'Partially disagree':
        numeric_option = -2
    elif option == 'Slightly disagree':
        numeric_option = -1
    elif option == 'neutral':
        numeric_option = 0
    elif option == 'Slightly Agree':
        numeric_option = 1
    elif option == 'Partially Agree':
        numeric_option = 2
    elif option == 'Fully Agree':
        numeric_option = 3
    else:
        numeric_option = 0  # Handle unexpected cases
    
    responses[f"Q{i+1}"] = numeric_option

# Button to trigger prediction


if st.button("Submit"):
    # Prepare data for the ML model
    X = np.array(list(responses.values())).reshape(1, -1)
    
    
    prediction = pt_model.predict([X])
    prediction = np.array(prediction)

   
    predicted_class_index = np.argmax(prediction)
    
    
    original_classes = range(0, 16)  # Assuming you have 13 classes
    predicted_class = original_classes[predicted_class_index]
    
    if predicted_class == 0:
        per = 'ESTJ'
        description = 'Efficient, organized, and decisive. They value tradition and order, and excel in leadership roles.'
    elif predicted_class == 1:
        per = 'ENTJ'
        description = 'Strategic, confident, and decisive. They are natural leaders who excel at setting goals and leading teams.'
    elif predicted_class == 2:
        per = 'ESFJ'
        description = 'Warm, caring, and sociable. They are sensitive to the needs of others and enjoy taking care of people.'
    elif predicted_class == 3:
        per = 'ENFJ'
        description = 'Charismatic, empathetic, and persuasive. They are natural leaders who inspire and motivate others.'
    elif predicted_class == 4:
        per = 'ISTJ'
        description = 'Practical, responsible, and orderly. They are known for their reliability and dedication to duty.'
    elif predicted_class == 5:
        per = 'ISFJ'
        description = 'Caring, supportive, and dependable. They are conscientious and enjoy providing practical support to others.'
    elif predicted_class == 6:
        per = 'INTJ'
        description = 'Analytical, innovative, and independent. They are deep thinkers who value knowledge and strategic planning.'
    elif predicted_class == 7:
        per = 'INFJ'
        description = 'Insightful, compassionate, and visionary. They are introspective and empathetic idealists.'
    elif predicted_class == 8:
        per = 'ESTP'
        description = 'Energetic, action-oriented, and adaptable. They enjoy living in the moment and taking risks.'
    elif predicted_class == 9:
        per = 'ESFP'
        description = 'Spontaneous, enthusiastic, and friendly. They love new experiences and thrive in social situations.'
    elif predicted_class == 10:
        per = 'ENTP'
        description = 'Innovative, curious, and outspoken. They are natural debaters who enjoy exploring new ideas and possibilities.'
    elif predicted_class == 11:
        per = 'ENFP'
        description = 'Enthusiastic, creative, and sociable. They are passionate about their beliefs and enjoy inspiring others.'
    elif predicted_class == 12:
        per = 'ISTP'
        description = 'Logical, practical, and adaptable. They excel at finding practical solutions to complex problems.'
    elif predicted_class == 13:
        per = 'ISFP'
        description = 'Sensitive, gentle, and compassionate. They value harmony and enjoy expressing themselves through artistic pursuits.'
    elif predicted_class == 14:
        per = 'INTP'
        description = 'Analytical, curious, and imaginative. They are independent thinkers who enjoy exploring ideas and theories.'
    elif predicted_class == 15:
        per = 'INFP'
        description = 'Idealistic, empathetic, and creative. They are deeply in tune with their values and strive to make a positive difference in the world.'
 
    # Display prediction
    st.write(f"### Predictrd Personality is {per}")
    #st.write(f"The predicted output based on responses is: {per}")
    st.write(f"{description}")
    
    


