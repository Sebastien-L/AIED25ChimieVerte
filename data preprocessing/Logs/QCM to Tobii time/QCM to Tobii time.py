import pandas as pd

# Define the column names we care about
start_pos_col = "start position seconds"
starttobii_col = "starttobii"
result_col = "result"

#these are the QCM times
#start_positions_qcm_2023 = [48,72,98,129,157,176,181,183,196,205,217,329,302,328,335,230,397,280,345]
#end_positions_qcm_2023 = [51,75,103,134,162,179,189,189,199,213,225,335,313,338,341,236,402,289,351]
start_positions_qcm_2024 = [48,72,98,129,157,176,181,196,205,217,329,302,335,230,397]
end_positions_qcm_2024 = [51, 75, 103, 134, 162, 179, 189, 199, 213, 225, 225, 313, 341, 236, 402]
start_positions_selfattention_2024 = [51,64,93,129,146,152,167,174,181,196,206,230,302,315,337,381,396,408] 
end_positions_selfattention_2024 = [60,67,101,135,150,161,173,178,194,199,213,235,312,321,341,394,402,417]

form_times = {
    "QCM_start_pos" : start_positions_qcm_2024,
    "QCM_end_pos" : end_positions_qcm_2024,
    "SelfReport_start_pos" : start_positions_selfattention_2024,
    "SelfReport_end_pos" : end_positions_selfattention_2024
}

def find_nearest_pos_in_panopto(ref_start_positions, start_pos):
    nearest_pos = ref_start_positions.max()
    for pos in ref_start_positions:
        if pos <= start_pos and abs(start_pos - pos) < abs(start_pos - nearest_pos):
            nearest_pos = pos
    return nearest_pos

# Loop over the 25 files
uids = [i for i in range(6,15)] + [i for i in range(16,30)]

for i in uids:
    #load the file
    filename = f"new panopto/panopto_data_updated_P24_{i:02}.csv"
    df = pd.read_csv(filename)
    result_df = pd.DataFrame(columns=df.columns)
    
    #QCM
    # Find the nearest smallest number and calculate the differences
    ref_start_positions = df[start_pos_col]
    for j in range(len(start_positions_qcm_2024)):
        nearest_pos = find_nearest_pos_in_panopto(ref_start_positions, start_positions_qcm_2024[j])
        temp_df = df.loc[df[start_pos_col] == nearest_pos].copy()
        temp_df["QCM_start_pos"] = (start_positions_qcm_2024[j] - nearest_pos) * 1000000 + temp_df[starttobii_col]
        
        nearest_pos = find_nearest_pos_in_panopto(ref_start_positions, end_positions_qcm_2024[j])
        temp_df = df.loc[df[start_pos_col] == nearest_pos].copy()
        temp_df["QCM_end_pos"] = (end_positions_qcm_2024[j] - nearest_pos) * 1000000 + temp_df[starttobii_col]
        
        #temp_df[result_col] /= 1000000
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # Save the result to a new CSV file
    output_filename = f"QCM to Tobii time/P24_QCM_{i:02}_output.csv"
    result_df.to_csv(output_filename, index=False)
    
    result_df = pd.DataFrame(columns=df.columns)
    
    
    #Self reports
    # Find the nearest smallest number and calculate the differences
    ref_start_positions = df[start_pos_col]
    for j in range(len(start_positions_selfattention_2024)):
        nearest_pos = find_nearest_pos_in_panopto(ref_start_positions, start_positions_selfattention_2024[j])
        temp_df = df.loc[df[start_pos_col] == nearest_pos].copy()
        temp_df["SelfReport_start_pos"] = (start_positions_selfattention_2024[j] - nearest_pos) * 1000000 + temp_df[starttobii_col]
        
        nearest_pos = find_nearest_pos_in_panopto(ref_start_positions, end_positions_selfattention_2024[j])
        temp_df = df.loc[df[start_pos_col] == nearest_pos].copy()
        temp_df["SelfReport_end_pos"] = (end_positions_selfattention_2024[j] - nearest_pos) * 1000000 + temp_df[starttobii_col]
        
        #temp_df[result_col] /= 1000000
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # Save the result to a new CSV file
    output_filename = f"QCM to Tobii time/P24_SelfReports_{i:02}_output.csv"
    result_df.to_csv(output_filename, index=False)
    
    result_df = pd.DataFrame(columns=df.columns)