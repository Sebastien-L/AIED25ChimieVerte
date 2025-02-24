import csv
import pandas as pd 
import os

def parse_coordinates(spatial_coordinates, videosize, screensize):
	#turn into python list
	coords = spatial_coordinates[1:-1].split(" ")
	shape = coords.pop(0) 
	
	if shape == "2": # Rectangle. Format: [x y l h]
		coords[0] = round(float(coords[0]) * (screensize[0] / videosize[0] * 1.0)) # x1
		coords[1] = round(float(coords[1]) * (screensize[1] / videosize[1] * 1.0)) # y1
		coords[2] = round(float(coords[2]) * (screensize[0] / videosize[0] * 1.0)) # x2
		coords[3] = round(float(coords[3]) * (screensize[1] / videosize[1] * 1.0)) #y2
	
		x1 = coords[0]
		x2 = coords[0] + coords[2]
		y1 = coords[1]
		y2 = coords[1] + coords[3]
		return str(x1)+","+str(y1)+"\t"+ \
				str(x1)+","+str(y2)+"\t"+ \
				str(x2)+","+str(y2)+"\t"+ \
				str(x2)+","+str(y1)
				
	elif shape == "3": #circle. Format: [x y r] #not supported by EMDAT, convert to square
		x1 = float(coords[0]) - float(coords[2])
		x2 = float(coords[0]) + float(coords[2])
		y1 = float(coords[1]) - float(coords[2])
		y2 = float(coords[1]) + float(coords[2])
		
		x1 = round(x1 * (screensize[0] / videosize[0] * 1.0)) # x1
		y1 = round(y1 * (screensize[1] / videosize[1] * 1.0)) # y1
		x2 = round(x2 * (screensize[0] / videosize[0] * 1.0)) # x2
		y2 = round(y2 * (screensize[1] / videosize[1] * 1.0)) # y2
		
		return str(x1)+","+str(y1)+"\t"+ \
				str(x1)+","+str(y2)+"\t"+ \
				str(x2)+","+str(y2)+"\t"+ \
				str(x2)+","+str(y1)
				
	if shape == "6" or shape == "7": # Polygon. Format: [x y x y x y...]
		ltemp = ""
		for i in range(0, len(coords), 2):
			coords[i] = round(float(coords[i]) * (screensize[0] / videosize[0] * 1.0)) # x1
			coords[i+1] = round(float(coords[i+1]) * (screensize[1] / videosize[1] * 1.0)) # y1
			
			if ltemp != "": ltemp += "\t"
			ltemp += str(coords[i]) + "," + str(coords[i+1])
		return ltemp
		
	else:
		print("shape {} not supported".format(shape))
		return ""


def find_nearest_pos_in_panopto(ref_start_positions, start_pos):
	nearest_pos = ref_start_positions.max()
	for pos in ref_start_positions:
		if pos <= start_pos and abs(start_pos - pos) < abs(start_pos - nearest_pos):
			nearest_pos = pos
	return nearest_pos

def video_to_tobii_time(uid, time_s):
	#time_ms = time_s * 1000000 #microseconds
	
	filename = f"../Logs/new panopto/panopto_data_updated_P24_{uid:02}.csv"
	df = pd.read_csv(filename)
	
	#QCM
	# Find the nearest smallest number and calculate the differences
	ref_start_positions = df["start position seconds"]
	
	#start
	nearest_pos = find_nearest_pos_in_panopto(ref_start_positions, time_s)
	temp_df1 = df.loc[df["start position seconds"] == nearest_pos].copy()
	res = (time_s - nearest_pos) * 1000000 + temp_df1["starttobii"]
	#print(time_s, "----", nearest_pos, "--**--", temp_df1["starttobii"])
		
	return int(res)
		
		
		
	

##############################################################################################
videosize = (960, 540)
screensize = (1280, 1024)
aoidef_file = "AOIs_definition.csv"

uids = [i for i in range(1,15)] + [i for i in range(16,30)]

#Detailed AOIs
outdir = "useraoi_detailed/"
outdirtype = "useraoi_types/"
outdirimp = "useraoi_importance/"
outdirtypeimp = "useraoi_type_importance/"

if not os.path.exists(outdir): os.makedirs(outdir)
if not os.path.exists(outdirtype): os.makedirs(outdirtype)
if not os.path.exists(outdirimp): os.makedirs(outdirimp)
if not os.path.exists(outdirtypeimp): os.makedirs(outdirtypeimp)

for u in uids:
	uid = "P24_0"+str(u) if u < 10 else "P24_"+str(u)
	outdata = {}
	typedata = {}
	impdata = {}
	typeimpdata = {}
	
	fout = open(outdir+uid+"_dynamicaoi.aoi", "w")
	fouttype = open(outdirtype+uid+"_dynamicaoi.aoi", "w")
	foutimp = open(outdirimp+uid+"_dynamicaoi.aoi", "w")
	fouttypeimp = open(outdirtypeimp+uid+"_dynamicaoi.aoi", "w")
	
	with open(aoidef_file, 'r') as faoi:
		reader = csv.DictReader(faoi)
		
		for row in reader:
			aoiname = str(row["ID"])
			aoi_coords = parse_coordinates( row["coordinates"], videosize, screensize )
			aoi_start_time = video_to_tobii_time(u, float(row["start"]))
			aoi_end_time = video_to_tobii_time(u, float(row["end"]))
			type_aoi = str(row["type"])
			imp_aoi = str(row["importance"])
			type_imp_aoi = type_aoi+"_"+imp_aoi
			
			outshape = "\t" + aoi_coords + "\n"
			outtimes = "#\t"+str(round(aoi_start_time))+","+str(round(aoi_end_time))  + "\n"
			
			if aoiname not in outdata: outdata[aoiname] = []
			if type_aoi not in typedata: typedata[type_aoi] = []
			if imp_aoi not in impdata: impdata[imp_aoi] = []
			if type_imp_aoi not in typeimpdata: typeimpdata[type_imp_aoi] = []
			
			outdata[aoiname].append( (outshape, outtimes) )
			typedata[type_aoi].append( (outshape, outtimes) )
			impdata[imp_aoi].append( (outshape, outtimes) )
			typeimpdata[type_imp_aoi].append( (outshape, outtimes) )
			
		#write
		for k,v in outdata.items():	
			for shape, times in v:
				fout.write( k+shape ) #aoi shape
				fout.write( times ) #aoi time
				
		for k,v in typedata.items():	
			for shape, times in v:
				fouttype.write( k+shape ) #aoi shape
				fouttype.write( times ) #aoi time
				
		for k,v in impdata.items():	
			for shape, times in v:
				foutimp.write( k+shape ) #aoi shape
				foutimp.write( times ) #aoi time

		for k,v in typeimpdata.items():	
			for shape, times in v:
				fouttypeimp.write( k+shape ) #aoi shape
				fouttypeimp.write( times ) #aoi time


####################################################################
#baseline static AOI

def static_aoi_grid(screensize, size = 3, dirstatic = "staticaoi/"):
	w = round(screensize[0]/size)
	h = round(screensize[1]/size)
	with open(dirstatic+f"{size}x{size}_staticaoi.aoi", 'w') as faoi:
		shape=""
		for i in range(size):
			for j in range(size):
				shape += str(i)+"_"+str(j)+"\t"+str(i*w)+","+str(j*h)+ \
						 "\t"+str((i+1)*w)+","+str(j*h)+ \
						 "\t"+str((i+1)*w)+","+str((j+1)*h)+ \
						 "\t"+str(i*w)+","+str((j+1)*h)+"\n"
		faoi.write(shape)


static_aoi_grid(screensize, 3, "staticaoi/") # 3x3
static_aoi_grid(screensize, 4, "staticaoi/") # 4x4

