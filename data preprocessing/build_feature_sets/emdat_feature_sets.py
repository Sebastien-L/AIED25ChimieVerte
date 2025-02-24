import pandas as pd
import numpy as np

windows = [10, 15, 20, 25, 30, "Content"]
targets = ["QCM", "SelfReports"]
studyyear = "23" #"24"
aois = ["useraoi_importance", "useraoi_types", "useraoi_type_importance", "useraoi_detailed", "staticAOI4x4", "staticAOI3x3", "staticAOI2x1", "no_aoi"]
outdir = "Features_AOI/20"+str(studyyear)

#read labels
qcm_labels = pd.read_csv("../../../Data/Data20"+str(studyyear)+"/Forms/QCM_Labels.csv", delimiter=',')
sr_labels = pd.read_csv("../../../Data/Data20"+str(studyyear)+"/ChimieVerte_forms/SelfReport_Labels.csv", delimiter=',')
qcm_labels['Answer'] =qcm_labels['Answer'].astype(int)
sr_labels['Answer'] = sr_labels['Answer'].astype(int)

for t in targets:
    #segments with a fixed window size, a set in the windows list
    for window in windows:
        for aoi in aois:
            # Read the TSV file
            if aoi == "no_aoi":
                data = pd.read_csv(f"../emdat/src/output/20"+str(studyyear)+"/output_no_AOI/chimieverte_features_Window_"+str(t)+"_"+str(window)+".tsv", delimiter='\t')
            else:
                data = pd.read_csv(f"../emdat/src/output/20"+str(studyyear)+"/chimieverte_features_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", delimiter='\t')
            
            data['Part_id'] = np.where(data['Part_id'] < 10, "P"+str(studyyear)+"_0" + data['Part_id'].astype(str), "P"+str(studyyear)+"_"+ data['Part_id'].astype(str))
            data['Part_id'] = data['Part_id'].astype(str)

            # Remove rows with 'allsc' in the 'Sc_id' column
            data = data[~data['Sc_id'].str.contains('allsc')]
            
            print(data.head())
            print(qcm_labels.head())
            
            #merge with labels
            if t == "QCM":
                data = pd.merge(data, qcm_labels, on = ["Part_id", "Sc_id"], how ='inner')
            else:
                data = pd.merge(data, sr_labels, on = ["Part_id", "Sc_id"], how ='inner')

            # Remove columns containing 'blink', 'saccade', or 'sum' in their names
            data = data.drop(columns=data.columns[data.columns.str.contains('blink|saccade|sum|event|clic')])

            # Remove rows where 'answer' column is null
            data = data.dropna(subset=['Answer'])


            if t == "QCM":
                # Replace values between 0 and 1 (exclusive) in 'answer' column with 0
                data.loc[(data['Answer'] > 0) & (data['Answer'] < 1), 'Answer'] = 0
            else:
                # Median split
                data.loc[data['Answer'] < 4, 'Answer'] = 0
                data.loc[data['Answer'] >= 4, 'Answer'] = 1


            # Remove 'numsegments'
            columns_to_remove = ['numsegments', 'largest_data_gap', 'length', 'length_invalid', 'numsamples', 'proportion_valid_fix', 'numfixations']
            data = data.drop(columns=columns_to_remove)

            # Save the modified data to a new TSV file
            data.to_csv(outdir+"modified_chimieverte_ETfeatures_All_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", sep='\t', index=False)

            #-- AOI only feature sets
            if aoi != "no_aoi":
                # Filter AOI columns only
                fixation_columns = [col for col in data.columns if '_' in col] #AOI featureq are the only one with "_" in their name, i.e., AOINAME_feature
                # Columns to keep
                columns_to_keep = ['Answer'] + fixation_columns
                # Keep only the desired columns
                data_filtered = data[columns_to_keep]
                # Save the resulting dataframe as a new TSV file
                data_filtered.to_csv(outdir+"modified_chimieverte_ETfeatures_AOIonly_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", sep='\t', index=False)

            #----
            # NO AOI sub feature sets
            if aoi == "no_aoi":

                # Filter all non-AOI columns
                fixation_columns = [col for col in data.columns if '_' not in col] #AOI featureq are the only one with "_" in their name, i.e., AOINAME_feature
                # Columns to keep
                columns_to_keep = ['Part_id', 'Sc_id'] + fixation_columns
                # Keep only the desired columns
                data_filtered = data[columns_to_keep]
                # Save the resulting dataframe as a new TSV file
                data_filtered.to_csv(outdir+"modified_chimieverte_ETfeatures_nonAOIonly_Window_"+str(t)+"_"+str(window)+".tsv", sep='\t', index=False)

                # --
                #fixation features only
                # Filter columns with 'fixation' in their names
                fixation_columns = [col for col in data.columns if 'fixation' in col]

                # Columns to keep
                columns_to_keep = ['Part_id', 'Sc_id', 'Answer'] + fixation_columns

                # Keep only the desired columns
                data_filtered = data[columns_to_keep]

                # Save the resulting dataframe as a new TSV file
                data_filtered.to_csv(outdir+"modified_chimieverte_ETfeatures_fixations_Window_"+str(t)+"_"+str(window)+".tsv", sep='\t', index=False)
                
                # --
                #fixation features only
                # Filter columns with 'fixation' in their names
                path_columns = [col for col in data.columns if 'path' in col]

                # Columns to keep
                columns_to_keep = ['Part_id', 'Sc_id', 'Answer'] + path_columns

                # Keep only the desired columns
                data_filtered = data[columns_to_keep]

                # Save the resulting dataframe as a new TSV file
                data_filtered.to_csv(outdir+"modified_chimieverte_ETfeatures_path_Window_"+str(t)+"_"+str(window)+".tsv", sep='\t', index=False)
                
                # --
                #fixation features only
                # Filter columns with 'fixation' in their names
                pupil_columns = [col for col in data.columns if 'pupil' in col]

                # Columns to keep
                columns_to_keep = ['Part_id', 'Sc_id', 'Answer'] + pupil_columns

                # Keep only the desired columns
                data_filtered = data[columns_to_keep]

                # Save the resulting dataframe as a new TSV file
                data_filtered.to_csv(outdir+"modified_chimieverte_ETfeatures_pupil_Window_"+str(t)+"_"+str(window)+".tsv", sep='\t', index=False)
                
                # --
                #fixation features only
                # Filter columns with 'fixation' in their names
                distance_columns = [col for col in data.columns if 'distance' in col]
                distance_columns = [col for col in distance_columns if 'path' not in col]

                # Columns to keep
                columns_to_keep = ['Part_id', 'Sc_id', 'Answer'] + distance_columns

                # Keep only the desired columns
                data_filtered = data[columns_to_keep]

                # Save the resulting dataframe as a new TSV file
                data_filtered.to_csv(outdir+"modified_chimieverte_ETfeatures_headdistance_Window_"+str(t)+"_"+str(window)+".tsv", sep='\t', index=False)