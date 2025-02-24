import json, re
"""
Origin: Top left
Video size: 960 x 540
# Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)
# Notes:
# - spatial_coordinates of [2,10,20,50,80] denotes a rectangle (shape_id=2) of size 50x80 placed at (10,20)
# - temporal coordinate of [1.349,2.741] denotes a temporal segment from time 1.346 sec. to 2.741 sec.
# - temporal coordinate of [4.633] denotes a still video frame at 4.633 sec.
# - metadata of {""1"":""3""} indicates attribute with id "1" set to an attribute option with id "3"
# SHAPE_ID = {"POINT":1,"RECTANGLE":2,"CIRCLE":3,"ELLIPSE":4,"LINE":5,"POLYLINE":6,"POLYGON":7,"EXTREME_RECTANGLE":8,"EXTREME_CIRCLE":9}
# FLAG_ID = {"RESERVED_FOR_FUTURE":1}
# ATTRIBUTE = {"1":{"aname":"Comment","anchor_id":"FILE1_Z2_XY0","type":1,"desc":"Free Comment","options":{"default":"Default"},"default_option_id":""},"3":{"aname":"Scene","anchor_id":"FILE1_Z2_XY0","type":1,"desc":"Video part","options":{},"default_option_id":""},"4":{"aname":"Type","anchor_id":"FILE1_Z1_XY1","type":3,"desc":"Type of AOI","options":{"0":"Title","1":"Chart","2":"Formula","3":"Molecule","4":"Picture","5":"Text","6":"Other"},"default_option_id":"6"},"5":{"aname":"Importance","anchor_id":"FILE1_Z1_XY1","type":3,"desc":"Importance","options":{"0":"High","1":"Med","2":"Low"},"default_option_id":"1"}}
# CSV_HEADER = metadata_id,file_list,flags,temporal_coordinates,spatial_coordinates,metadata
"1_j5sJvB9h","[""chimie-verte_version-française_default.mp4""]",0,"[1.124,5.31493]","[]","{""1"":""_DEFAULT"",""3"":""main title""}"
"1_zYQJk2sM","[""chimie-verte_version-française_default.mp4""]",0,"[2.41892]","[2,162.262,176.21971,639.45,192.49481]","{""1"":""AOI"",""4"":""0"",""5"":""2""}"
1_vL68gsEX	["chimie-verte_version-franÃ§aise_default.mp4"]	0	[0.0441]	[2,2.7551,2.7551,955.10204,534.4898]	{"1":"CALIBRATION","4":"6","5":"1"}																										

"""

class Segment:
	def __init__(self, start, end, name):
		self.name = name
		self.time_start = start
		self.time_end = end
	
	def __repr__(self):
		return " ".join([self.name,str(self.time_start), str(self.time_end)])
		
	def __str__(self):
		return " ".join([self.name,str(self.time_start), str(self.time_end)])
		
	def in_segment(self, time):
		return self.time_start <= time <= self.time_end
		
	def find_segment(seglist, time): #static function
		for seg in seglist:
			if seg.in_segment(time):
				return (seg.time_start, seg.time_end)

class AOI:
	def __init__(self, id, start, end, coordinates, type, importance):
		self.id = id
		self.time_start = start
		self.time_end = end
		self.coordinates = coordinates
		self.type = type
		self.importance = importance
		
	def __repr__(self):
		return " ".join([self.id,str(self.time_start), str(self.time_end), str(self.coordinates), self.type, self.importance])
		
	def __str__(self):
		return " ".join([self.id,str(self.time_start), str(self.time_end), str(self.coordinates), self.type, self.importance])
		
	def to_csv(self, sep=","):
		return sep.join([self.id,str(self.time_start), str(self.time_end), str(self.coordinates).replace(",", ""), self.type, self.importance])

	def csv_header(sep=","):  #static function
		return sep.join(["ID", "start", "end", "coordinates", "type", "importance"])
		sep.join([self.id,str(self.time_start), str(self.time_end), str(self.coordinates).replace(",", ""), self.type, self.importance])
		
	def parse_row(row, sep=","): #static function
		row = re.sub(r'(?<!")"(?!")', "", row.strip()) # remove single quotes
		row = row.replace("\"\"", "\"") # turn double quotes into single one, as in Python
		
		out = []
		ctemp = ""
		nested_start = ["[", "{"]
		nested_end = ["]", "}"]
		nested = 0
		
		for c in row:
			if c == sep and nested == 0:
				out.append(ctemp)
				ctemp = ""
			else:
				if c in nested_start:
					nested += 1
				if c in nested_end:
					nested -= 1
				ctemp += c
		out.append(ctemp)
		return out
		
			
header_tags = { "csvheader": "CSV_HEADER", "annotation_template": "ATTRIBUTE" }
template_tags = { "type": "Type", "importance": "Importance" }
aoi_valid_tag_name = "AOI"
segments_list = []
header = ""
template_types = []
template_importance = []
AOI_list = []
VIA_fileexport = "ChimieVerteAOIs10Nov2024_23h45m28s_export.csv"
outfile = "AOIs_definition.csv"

#extract segment first (not in order so need to parse the full file first)
with open(VIA_fileexport, "r") as f:
    for l in f:
        if l.startswith("#"):
            if header_tags["csvheader"] in l:
                header = l.split("=")[1].strip()
                header = header.split(",")
            if header_tags["annotation_template"] in l:
                template = json.loads(l.split("=")[1].strip())
                for k,v in template.items():
                    if template_tags["type"] in v["aname"]:
                        template_types = (k, v["options"])
                    if template_tags["importance"] in v["aname"]:
                        template_importance = (k, v["options"])

        else:
            row = {i : j for i, j in zip(header, AOI.parse_row(l))}
            #print(row)
            timestamps = eval(row["temporal_coordinates"])
            annotations = eval(row["metadata"])

            #check if segment
            if len(timestamps) > 1 and len(annotations) > 1: #segment with start and end time
                seg_name = annotations["3"]
                if seg_name:
                    segments_list.append(Segment(timestamps[0], timestamps[1], annotations["3"]))

print(template_types)
print(template_importance)
print(segments_list)

                
#extract AOIs now (again not in order)
with open(VIA_fileexport, "r") as f:
    for l in f:
        if not l.startswith("#"):
            row = {i : j for i, j in zip(header, AOI.parse_row(l))}
            #print(row)
            timestamps = eval(row["temporal_coordinates"])
            coordinates = eval(row["spatial_coordinates"])
            annotations = eval(row["metadata"])

            #check if AOI
            if len(timestamps) == 1 and len(coordinates) > 0 and annotations["1"] == aoi_valid_tag_name: #AOI with coodinates
                #print( template_types[1][str(annotations[str(template_types[0])])] )
                (start, end) = Segment.find_segment(segments_list, timestamps[0])
                if "6" not in annotations: print(annotations)
                aoiname = annotations["6"]
                AOI_list.append(AOI( aoiname, start, end, coordinates, template_types[1][str(annotations[str(template_types[0])])], template_importance[1][str(annotations[str(template_importance[0])])] ))


print(AOI_list)
print(len(AOI_list))

with open (outfile, "w") as f:
	f.write(AOI.csv_header()+"\n")
	for aoi in AOI_list:
		f.write(aoi.to_csv()+"\n")

