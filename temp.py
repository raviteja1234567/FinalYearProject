# -*- coding: utf-8 -*-
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

rest=0
exer=0
c1=0
c2=0

if(age>=18 and age<=35):
        while(c1<5):
            mydata=int(ardData.readline().strip())
            if(mydata>=49 and mydata<=82):
                print(mydata)
                rest+=mydata
                c1=c1+1
        rest=rest/5
        print("RESTING HEART RATE")
        print(rest)
        firebase.put('','value',rest)   
elif(age>=36 and age<=55):
        while(c1<5):
            mydata=int(ardData.readline().strip())
            if(mydata>=50 and mydata<=84):
                print(mydata)
                rest+=mydata
                c1=c1+1
        rest=rest/5
        print("RESTING HEART RATE")
        print(rest)
        firebase.put('','value',rest)           
elif(age>=56 and age<=70):
        while(c1<5):
            mydata=int(ardData.readline().strip())
            if(mydata>=51 and mydata<=82):
                print(mydata)
                rest+=mydata
                c1=c1+1           
        rest=rest/5
        print("RESTING HEART RATE")
        print(rest)
        firebase.put('','value',rest)   

        
    
    

