# -*- coding: utf-8 -*-
"""
Created on Mon May 21 00:58:44 2018

@author: DEEPAK S
"""

"""
Spyder Editor

This is a temporary script file.
"""
from firebase import firebase
import serial
ardData=serial.Serial('COM3',9600)
firebase=firebase.FirebaseApplication('https://pulse-1e371.firebaseio.com/')
res=firebase.get('','age')
"""res1=firebase.get('','state')"""
age=int(res)

exer=0
c2=0
while(c2<5):
        mydata=int(ardData.readline().strip())
        if(mydata>=((220-age)*3/5) and mydata<=((220-age)*9/10)):
            print(mydata)
            exer+=mydata
            c2=c2+1
exer=exer/5
print("EXERCISING HEART RATE")
print(exer)
firebase.put('','value2',exer)
        
    
    


