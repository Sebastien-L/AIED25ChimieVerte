import sys, csv

windows = [10, 15, 20, 25, 30]
targets = ["QCM", "SelfReports"]

ferr = open("incorrect_generated_segments.txt", "w")

uids = [i for i in range(1,15)] + [i for i in range(16,30)]
#uids = [i for i in range(1,26)]

for pid in uids:
    #open participant CSV file with all the times computed from the Logs
    for t in targets:
        #segments with a fixed window size, a set in the windows list
        for window in windows:
            with open("../Logs/QCM to Tobii time/P24_"+str(t)+"_"+(str(pid) if pid > 9 else "0"+str(pid))+"_output.csv", "r") as qcm_time:
            #with open("../Logs/QCM to Tobii time/P23_"+str(t)+"_"+(str(pid) if pid > 9 else "0"+str(pid))+"_output.csv", "r") as qcm_time:
                qcm_time_dict = csv.DictReader(qcm_time)
                segid = 1
                with open("src/data_RecoveryDataEyeTracker/Recovery_Participant24_"+str(t)+"_"+str(pid)+"_Window_"+str(window)+".seg", "w") as seg_file:
                #with open("src/data_RecoveryDataEyeTracker/Recovery_Participant23_"+str(t)+"_"+str(pid)+"_Window_"+str(window)+".seg", "w") as seg_file:
                    for row in qcm_time_dict:
                        if row["pos"] != "":
                            if row["start_end"] == "end":
                                end_time = int(float(row["pos"]))
                                segout = "seg"+str(segid)+"\tseg"+str(segid)+"\t"+str(max(0, end_time - (window*1000000)))+"\t"+str(end_time)+"\n"
                                seg_file.write(segout)
                                segid += 1
                        else:
                            print("empty line for "+str(pid)+" segid="+str(segid)+" -- "+str(row))
                 
        with open("../Logs/QCM to Tobii time/P24_"+str(t)+"_"+(str(pid) if pid > 9 else "0"+str(pid))+"_output.csv", "r") as qcm_time:
        #with open("../Logs/QCM to Tobii time/P23_"+str(t)+"_"+(str(pid) if pid > 9 else "0"+str(pid))+"_output.csv", "r") as qcm_time:
            qcm_time_dict = csv.DictReader(qcm_time)
            #segments whose times are end-start, i.e., QCM_end_pos-QCM_start_pos
            with open("src/data_RecoveryDataEyeTracker/Recovery_Participant24_"+str(t)+"_"+str(pid)+"_Window_Content.seg", "w") as seg_file:
            #with open("src/data_RecoveryDataEyeTracker/Recovery_Participant23_"+str(t)+"_"+str(pid)+"_Window_Content.seg", "w") as seg_file:
                segid = 1
                start_time = 0
                for row in qcm_time_dict:
                    if row["pos"] != "":
                        if row["start_end"] == "start":
                            start_time = int(float(row["pos"]))
                        else:
                            end_time = int(float(row["pos"]))
                            if start_time == 0 or start_time >= end_time:
                                print("Warning: empty start time or superior to end time: P"+str(pid)+"_"+str(t)+"_Seg"+str(segid))
                                ferr.write("P"+str(pid)+"_"+str(t)+"_Seg"+str(segid)+"\n")
                            else:
                                segout = "seg"+str(segid)+"\tseg"+str(segid)+"\t"+str(start_time)+"\t"+str(end_time)+"\n"
                                seg_file.write(segout)
                            segid += 1
                    else:
                        print("empty line for "+str(pid)+" segid="+str(segid)+" -- "+str(row))

ferr.close()                   
print("done")