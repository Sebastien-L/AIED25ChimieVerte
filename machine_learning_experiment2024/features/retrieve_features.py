import pandas as pd

def merge_features(path23, path24, output_file):
    # Load the DataFrames
    df_24 = pd.read_csv(path24, sep='\t')
    df_23 = pd.read_csv(path23, sep='\t')
    
    # Check if columns match
    if list(df_24.columns) == list(df_23.columns):
        # Merge the DataFrames
        merged_df = pd.concat([df_24, df_23], ignore_index=True)
        
        # Save the merged DataFrame
        merged_df.to_csv(output_file, index=False)
        print(output_file, merged_df["Part_id"].nunique(), merged_df.shape[0], len(merged_df.columns), sep=",")
    else:
        print(f"------ Columns do not match for {output_file}. Merge operation aborted.")
        cols_in_24_not_23 = set(df_24.columns) - set(df_23.columns)
        cols_in_23_not_24 = set(df_23.columns) - set(df_24.columns)
        
        """if cols_in_24_not_23:
            print(f"Columns in 2024 but not in 2023: {list(cols_in_24_not_23)}")
        if cols_in_23_not_24:
            print(f"Columns in 2023 but not in 2024: {list(cols_in_23_not_24)}")"""


# Define prefixes and common file names
prefixes = ["24", "23"]  # Two series of DataFrames
dir23 = "../../data preprocessing/build_feature_sets/Features_AOI_2023/"
dir24 = "../../data preprocessing/build_feature_sets/Features_AOI_2024/"
output_prefix = "merged_"  # Prefix for the merged files

windows = [10, 15, 20, 25, 30, "Content"]
targets = ["QCM", "SelfReports"]
aois = ["useraoi_importance", "useraoi_types", "useraoi_type_importance", "useraoi_detailed","staticAOI2x1"]

for t in targets:
    #segments with a fixed window size, a set in the windows list
    for window in windows:
        for aoi in aois:
            # ET all
            merge_features(dir23+"modified_chimieverte_ETfeatures_All_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", dir24+"modified_chimieverte_ETfeatures_All_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", "MLfeatures_ET_All_"+str(t)+"_"+str(window)+'_'+str(aoi)+".csv")

            # ET AOI only
            merge_features(dir23+"modified_chimieverte_ETfeatures_AOIonly_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", dir24+"modified_chimieverte_ETfeatures_AOIonly_Window_"+str(t)+"_"+str(window)+'_'+str(aoi)+".tsv", "MLfeatures_ET_AOIonly_"+str(t)+"_"+str(window)+'_'+str(aoi)+".csv")

        # ET noAOI
        merge_features(dir23+"modified_chimieverte_ETfeatures_nonAOIonly_Window_"+str(t)+"_"+str(window)+".tsv", dir24+"modified_chimieverte_ETfeatures_nonAOIonly_Window_"+str(t)+"_"+str(window)+".tsv", "MLfeatures_ET_noAOI_"+str(t)+"_"+str(window)+"_noAOI.csv")

        # ET fixation
        merge_features(dir23+"modified_chimieverte_ETfeatures_fixations_Window_"+str(t)+"_"+str(window)+".tsv", dir24+"modified_chimieverte_ETfeatures_fixations_Window_"+str(t)+"_"+str(window)+".tsv", "MLfeatures_ET_fixations_"+str(t)+"_"+str(window)+"_noAOI.csv")

        # ET saccades
        merge_features(dir23+"modified_chimieverte_ETfeatures_path_Window_"+str(t)+"_"+str(window)+".tsv", dir24+"modified_chimieverte_ETfeatures_path_Window_"+str(t)+"_"+str(window)+".tsv", "MLfeatures_ET_saccades_"+str(t)+"_"+str(window)+"_noAOI.csv")

        # ET pupil
        merge_features(dir23+"modified_chimieverte_ETfeatures_pupil_Window_"+str(t)+"_"+str(window)+".tsv", dir24+"modified_chimieverte_ETfeatures_pupil_Window_"+str(t)+"_"+str(window)+".tsv", "MLfeatures_ET_pupils_"+str(t)+"_"+str(window)+"_noAOI.csv")

        # ET head distance
        merge_features(dir23+"modified_chimieverte_ETfeatures_headdistance_Window_"+str(t)+"_"+str(window)+".tsv", dir24+"modified_chimieverte_ETfeatures_headdistance_Window_"+str(t)+"_"+str(window)+".tsv", "MLfeatures_ET_headdistance_"+str(t)+"_"+str(window)+"_noAOI.csv")