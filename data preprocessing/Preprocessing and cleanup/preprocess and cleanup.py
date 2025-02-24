import pandas as pd
import numpy as np 

dataBase = []

uids = [i for i in range(1,15)] + [i for i in range(16,30)]
for i in uids:
    pid = str(i) if i > 9 else '0'+str(i)
    readValue = pd.read_table(r"../../../Data/Data2024/ChimieVerte_DataExportTobii/LabelVert2024 Recording"+str(pid)+".tsv", sep='\t')
    dataBase.append(readValue)
 

timestamps_tab = pd.read_csv('../Logs/Participants_timestamps_2024.csv') 
startTask=[]
endTask=[]
for i in range(len(timestamps_tab)):
    
    tmpStart= "00:00:00.00" if type(timestamps_tab['Task start'][i]) == float  else timestamps_tab['Task start'][i]    
    hours, minutes, seconds = tmpStart.split(":")
    h = int(hours)
    m = int(minutes)
    s = float(seconds)
    ml = int(3600000000 * h + 60000000 * m + 1000000 * s)
     
    startTask.append(ml)

    tmpEnd= "00:00:00.00" if type( timestamps_tab['Task end'][i] )== float else timestamps_tab['Task end'][i]
    hours, minutes, seconds = tmpEnd.split(":")
    h = int(hours)
    m = int(minutes)
    s = float(seconds)
    ml = int(3600000000 * h + 60000000 * m + 1000000 * s)
    
    endTask.append(ml)
print(startTask)
print(endTask) 


dataBaseTask=[]

for i in range(len(timestamps_tab)):
    dataBaseTask.append(dataBase[i].loc[(dataBase[i]["Recording timestamp"] >=startTask[i] ) & (dataBase[i]["Recording timestamp"] <=endTask[i])])

dataBaseTask[0].tail() 


dataEyeTracker=[]

for i in range(len(timestamps_tab)):
    mask = dataBaseTask[i]['Sensor'].isin(['Eye Tracker'])
    dataEyeTracker.append(dataBaseTask[i].loc[mask])
dataEyeTracker[0].head() 


print("\n\n****************************** valid, invalid, Eyes Not found [In task] ***************************")

matrice=[]
for i in range(len(timestamps_tab)):
    
    indexEyesNotFound = dataEyeTracker[i][ dataEyeTracker[i]['Eye movement type'] =='EyesNotFound'].index
    indexUnclassified = dataEyeTracker[i][ dataEyeTracker[i]['Eye movement type'] ==  'Unclassified'].index
    fix = dataEyeTracker[i][ dataEyeTracker[i]['Eye movement type'] =='Fixation' ].index
    sac = dataEyeTracker[i][ dataEyeTracker[i]['Eye movement type'] =='Saccade' ].index
    
    indexInvalidLeft = dataEyeTracker[i][ dataEyeTracker[i]['Validity left'] == 'Invalid'].index
    indexInvalidRight = dataEyeTracker[i][ dataEyeTracker[i]['Validity right'] == 'Invalid'].index
    validLeft = dataEyeTracker[i][ dataEyeTracker[i]['Validity left'] == 'Valid'].index
    validRight = dataEyeTracker[i][ dataEyeTracker[i]['Validity right'] == 'Valid'].index
       
    matrice.append([len(indexEyesNotFound),
                    len(indexUnclassified),
                    len(fix)+len(sac),
                    str(int(((len(fix)+len(sac)) /len(dataEyeTracker[i]))*100))+" %",
                    len(indexInvalidLeft),
                    len(validLeft),
                    str(int((len(validLeft)/len(dataEyeTracker[i]))*100))+" %",
                    len(indexInvalidRight),
                    len(validRight),
                    str(int((len(validRight)/len(dataEyeTracker[i]))*100))+" %",
                    len(dataEyeTracker[i])])

columns = ['EyesNotFound ','Unclassified','faxation & saccade','% (Fix,Sac)','InvalidLeft','ValidLeft','% valid L','InvalidRight','ValidRight','% valid R','Total Rows']
index=["participant "+str(i) for i in uids]

df = pd.DataFrame(data=matrice, index=index, columns=columns)
print(df)


dataAllParticipants=[]

for i in range(len(timestamps_tab)):

    dataAllMinute=[]

    pointDepart=dataEyeTracker[i].head(1).iloc[0]['Recording timestamp']
    endOfTime=dataEyeTracker[i].tail(1).iloc[0]['Recording timestamp']
    
    j=0
    while j < dataEyeTracker[i].index[-1]:
    
        pft=dataEyeTracker[i][dataEyeTracker[i]['Recording timestamp']>=pointDepart+60000000] #1 minute

        if pft.empty :
            pointFin =endOfTime
        else:
            pointFin=pft.head(1).iloc[0]['Recording timestamp']

        mask=(dataEyeTracker[i]['Recording timestamp']>=pointDepart)&( dataEyeTracker[i]['Recording timestamp']<=pointFin )
        dataInMinute=dataEyeTracker[i].loc[mask]
        dataAllMinute.append(dataInMinute)

        pointDepart=pointFin
        j=dataEyeTracker[i][dataEyeTracker[i]['Recording timestamp']==pointFin].index[0]
        
    dataAllParticipants.append(dataAllMinute)

print(dataAllParticipants[-1][1].head() )


AllParticipantMinParMin=[]

for j in range(len(timestamps_tab)):
    matrice=[]
    for i in range(0,len(dataAllParticipants[j])):

        indexEyesNotFound = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Eye movement type'] =='EyesNotFound'].index
        indexUnclassified = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Eye movement type'] ==  'Unclassified'].index
        fix = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Eye movement type'] =='Fixation' ].index
        sac = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Eye movement type'] =='Saccade' ].index

        indexInvalidLeft = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Validity left'] == 'Invalid'].index
        indexInvalidRight = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Validity right'] == 'Invalid'].index
        validLeft = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Validity left'] == 'Valid'].index
        validRight = dataAllParticipants[j][i][ dataAllParticipants[j][i]['Validity right'] == 'Valid'].index

        matrice.append([len(indexEyesNotFound),
                        len(indexUnclassified),
                        len(fix)+len(sac),
                        str(int(((len(fix)+len(sac)) /len(dataAllParticipants[j][i]))*100))+" %",
                        len(indexInvalidLeft),
                        len(validLeft),
                        str(int((len(validLeft)/len(dataAllParticipants[j][i]))*100))+" %",
                        len(indexInvalidRight),
                        len(validRight),
                        str(int((len(validRight)/len(dataAllParticipants[j][i]))*100))+" %",
                        len(dataAllParticipants[j][i])])
    AllParticipantMinParMin.append(matrice)
   
print("\n****************************** valid , invalid , Eyes Not found [In task] ***************************\n")
n=3
print(" --Participant-- [ "+str(n),"]")
columns = ['EyesNotFound ','Unclassified','faxation & saccade','% (Fix,Sac)','InvalidLeft','ValidLeft','% valid L','InvalidRight','ValidRight','% valid R','Total Rows']
index=[]
for i in range(1,len(AllParticipantMinParMin[n-1])+1):
    index.append("Minute "+str(i))

df1 = pd.DataFrame(data=AllParticipantMinParMin[n-1],index=index,columns=columns)
print(df1)


tableMinParMin=[]

columns = ['EyesNotFound ','Unclassified','faxation & saccade','% (Fix,Sac)','InvalidLeft','ValidLeft','% valid L','InvalidRight','ValidRight','% valid R','Total Rows']

for i in range(len(timestamps_tab)):
    index=[]
    for j in range(1,len(AllParticipantMinParMin[i])+1):
        index.append("Minute "+str(j))
    tableMinParMin.append(pd.DataFrame(data=AllParticipantMinParMin[i],index=index,columns=columns))



z=1
for j in range(len(timestamps_tab)):
    
    mat=[]
    tmp=[]
    for i in dataEyeTracker[j].index:

        if ((dataEyeTracker[j].loc[[i],['Validity left']].values[0][0]=='Invalid') &
            (dataEyeTracker[j].loc[[i],['Validity right']].values[0][0]=='Invalid')):
            tmp.append(i)
        else :
            if len (tmp) >= 1 :
                mat.append(tmp)
                tmp=[]
    if len (tmp) >= 1 :
        mat.append(tmp)
    
    listitems =mat

    with open('InvalidDataIndex/participant_'+str(uids[j])+'.txt', 'w') as temp_file:
    
        for item in listitems:
            if item == listitems[-1]:
                temp_file.write("%s"%item)
            else:
                temp_file.write("%s,"%item)
                

    print("invalid mat participant",j,"completed ! ") 
    
    
    
InvalidMat=[]
tmpMat=[]
for i in uids:

    lstr=[]
    l=[]
    tmpMat=[]
    tmp=[]

    file = open('InvalidDataIndex/participant_'+str(i)+'.txt', 'r')
        
    lstr=file.read().split('],[')

    tmp=list(map(int,lstr[0][1:].split(',')))
    tmpMat.append(tmp)
    tmp=[]
    for j in range(1,len(lstr)-1):
        tmp=list(map(int,lstr[j].split(',')))
        tmpMat.append(tmp)
        tmp=[]
        
    tmp=list(map(int,lstr[-1][:-1].split(',')))
    tmpMat.append(tmp)
    tmp=[]
    
    InvalidMat.append(tmpMat)


print([len(k) for k in InvalidMat[-1]]) 
print([len(k) for k in mat]) 



RecoveryDataEyeTracker=[]
for j  in range(len(timestamps_tab)):
    copy = dataEyeTracker[j].copy()
    indice=copy.index
    listIndex=indice.values.tolist()


    for i in InvalidMat[j] :
        if (len(i) <=4 )& (len(i)>0) : 
            if( listIndex.index(i[0]) >0) & (listIndex.index(i[-1]) < len(listIndex)-1):
                #print(i)
                #print((copy.loc[[copy.loc[[listIndex[listIndex.index(i[0])-1]]].index[0]],['Eye movement type']].values[0][0]))
                #print((copy.loc[[copy.loc[[listIndex[listIndex.index(i[-1])+1]]].index[0]],['Eye movement type']].values[0][0]) )
                if ((copy.loc[[copy.loc[[listIndex[listIndex.index(i[0])-1]]].index[0]],['Eye movement type']].values[0][0]=='Saccade' )&
                   (copy.loc[[copy.loc[[listIndex[listIndex.index(i[-1])+1]]].index[0]],['Eye movement type']].values[0][0]=='Saccade') ):
                        
                       
                    for e in i:
                        copy.loc[[e],['Eye movement type']]='Saccade'
                        copy.loc[[e],['Validity left']]='Valid'
                        copy.loc[[e],['Validity right']]='Valid'

                if ((copy.loc[[copy.loc[[listIndex[listIndex.index(i[0])-1]]].index[0]],['Eye movement type']].values[0][0]=='Fixation' )&
                   (copy.loc[[copy.loc[[listIndex[listIndex.index(i[-1])+1]]].index[0]],['Eye movement type']].values[0][0]=='Fixation') ):
                        
                       

                    for e in i:
                        copy.loc[[e],['Eye movement type']]='Fixation'
                        copy.loc[[e],['Validity left']]='Valid'
                        copy.loc[[e],['Validity right']]='Valid'

                if ((copy.loc[[copy.loc[[listIndex[listIndex.index(i[0])-1]]].index[0]],['Eye movement type']].values[0][0]=='Fixation' )&
                   (copy.loc[[copy.loc[[listIndex[listIndex.index(i[-1])+1]]].index[0]],['Eye movement type']].values[0][0]=='Saccade') ):
                        
                       

                    for e in i:
                        copy.loc[[e],['Eye movement type']]='Saccade'
                        copy.loc[[e],['Validity left']]='Valid'
                        copy.loc[[e],['Validity right']]='Valid'

                if ((copy.loc[[copy.loc[[listIndex[listIndex.index(i[0])-1]]].index[0]],['Eye movement type']].values[0][0]=='Saccade' )&
                   (copy.loc[[copy.loc[[listIndex[listIndex.index(i[-1])+1]]].index[0]],['Eye movement type']].values[0][0]=='Fixation') ):
                        
                       
                    for e in i:
                        copy.loc[[e],['Eye movement type']]='Fixation'
                        copy.loc[[e],['Validity left']]='Valid'
                        copy.loc[[e],['Validity right']]='Valid'

    RecoveryDataEyeTracker.append(copy)    
    n=j+1    
    print("Data Recovery for Participant_"+str(n)+"    created ! ") 
    
    
for j in range(len(timestamps_tab)):
    RecoveryDataEyeTracker[j].to_csv("data_out/data_RecoveryDataEyeTracker/Recovery_Participant_"+str(uids[j])+".tsv", sep='\t')                                  
    print("data file  Recovery_Participant_"+str(uids[j])+".tsv    created ! ") 
    


print("\n\n****************************** valid , invalid , Eyes Not found [In Recovery data ] ***************************")

matriceRecovery=[]
for i in range(len(timestamps_tab)):
    
    indexEyesNotFound = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Eye movement type'] =='EyesNotFound'].index
    indexUnclassified = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Eye movement type'] ==  'Unclassified'].index
    fix = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Eye movement type'] =='Fixation' ].index
    sac = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Eye movement type'] =='Saccade' ].index
    
    indexInvalidLeft = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Validity left'] == 'Invalid'].index
    indexInvalidRight = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Validity right'] == 'Invalid'].index
    validLeft = RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Validity left'] == 'Valid'].index
    validRight =RecoveryDataEyeTracker[i][ RecoveryDataEyeTracker[i]['Validity right'] == 'Valid'].index
       
    matriceRecovery.append([len(indexEyesNotFound),
                    len(indexUnclassified),
                    len(fix)+len(sac),
                    str(int(((len(fix)+len(sac)) /len(RecoveryDataEyeTracker[i]))*100))+" %",
                    len(indexInvalidLeft),
                    len(validLeft),
                    str(int((len(validLeft)/len(RecoveryDataEyeTracker[i]))*100))+" %",
                    len(indexInvalidRight),
                    len(validRight),
                    str(int((len(validRight)/len(RecoveryDataEyeTracker[i]))*100))+" %",
                    len(RecoveryDataEyeTracker[i])])

columns = ['EyesNotFound ','Unclassified','faxation & saccade','% (Fix,Sac)','InvalidLeft','ValidLeft','% valid L','InvalidRight','ValidRight','% valid R','Total Rows']
index=[]
for i in uids:
    index.append("participant "+str(i))
    
#c='background-color: red' 

recoveryTable = pd.DataFrame(data=matriceRecovery,index=index,columns=columns)
print (recoveryTable )