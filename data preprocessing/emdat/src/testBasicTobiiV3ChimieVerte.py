"""
UBC Eye Movement Data Analysis Toolkit (EMDAT), Version 3
Created on 2015-08-15

Sample code to run EMDAT for a given experiment.
"""
import sys
from BasicParticipant import *
from EMDAT_core.Participant import export_features_all, write_features_tsv
from EMDAT_core.ValidityProcessing import output_Validity_info_Segments, output_percent_discarded, output_Validity_info_Participants


#read argument
if len(sys.argv) > 2:
    params.TARGET = str(sys.argv[1])
    params.WINDOW = str(sys.argv[2])
if len(sys.argv) == 4:
    params.AOI = str(sys.argv[3])
    aoisimplename = params.AOI.split("/")[-1]


# user list
ul = [i for i in range(1,15)] + [i for i in range(16,30)]
#ul = [8] #test


# user ids
uids = ul
# time offsets from start of the recording
alogoffset = [0]*len(ul)
alogoffset = [0]*len(ul)
params.AOI = "../../AOIs/staticaoi/2x1_staticaoi.aoi"
aoisimplename = "staticAOI2x1"

# Read participants
ps = read_participants_Basic(user_list = ul, pids = uids, log_time_offsets = alogoffset, datadir = params.EYELOGDATAFOLDER,
                             prune_length = None,
                             aoifile = params.AOI, #"../../AOIs/staticaoi/4x4_staticaoi.aoi",
                             require_valid_segs = False,
                             auto_partition_low_quality_segments = False)

if params.DEBUG or params.VERBOSE == "VERBOSE":
    # explore_validation_threshold_segments(ps, auto_partition_low_quality_segments = False)
    output_Validity_info_Segments(ps, auto_partition_low_quality_segments_flag = False, validity_method = 3)
    output_percent_discarded (ps, r'output/disc_Win'+str(params.TARGET)+"_"+str(params.WINDOW)+'.csv')
    output_Validity_info_Segments(ps, auto_partition_low_quality_segments_flag = False, validity_method = 2,threshold_gaps_list = [100, 200, 250, 300], output_file = "output/tobiiv3_Seg_val_Win"+str(params.TARGET)+"_"+str(params.WINDOW)+".csv" ) 
    output_Validity_info_Participants(ps, include_restored_samples = True, auto_partition_low_quality_segments_flag = False)


# WRITE features to file
#if params.VERBOSE != "QUIET":#
#    print#
#    print "Exporting:\n--General:", params.featurelist
#write_features_tsv(ps, './outputfolder/tobiiv3_sample_features_test.tsv', featurelist=params.featurelist, id_prefix=False)

aoi_feat_names = (map(lambda x:x, params.aoigeneralfeat))
if params.VERBOSE != "QUIET":
     print()
     print("Exporting features:\n--General:", params.featurelist, "\n--AOI:", params.aoifeaturelist)#, "\n--Sequences:", params.aoisequencefeat
#write_features_tsv(ps, r'output/chimieverte_features_Window_'+str(params.TARGET)+'_'+str(params.WINDOW)+'_'+str(aoisimplename)+'.tsv',featurelist = params.featurelist, aoifeaturelist = params.aoigeneralfeat, id_prefix = True)
#write_features_tsv(ps, r'output/chimieverte_aoisequences_Window_'+str(params.TARGET)+'_'+str(params.WINDOW)+'_'+str(aoisimplename)+'.tsv',featurelist = params.aoisequencefeat, aoifeaturelist=params.aoigeneralfeat, id_prefix = True)

write_features_tsv(ps, r'output/chimieverte_features_Window_'+str(params.TARGET)+'_'+str(params.WINDOW)+'_'+str(aoisimplename)+'.tsv',featurelist = params.featurelist, aoifeaturelabels = params.aoifeaturelist, id_prefix = True)
write_features_tsv(ps, r'output/chimieverte_aoisequences_Window_'+str(params.TARGET)+'_'+str(params.WINDOW)+'_'+str(aoisimplename)+'.tsv',featurelist = params.aoisequencefeat, aoifeaturelabels=params.aoifeaturelist, id_prefix = True)