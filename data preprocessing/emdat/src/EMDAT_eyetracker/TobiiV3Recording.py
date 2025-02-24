"""
UBC Eye Movement Data Analysis Toolkit (EMDAT), Version 3
Created on 2015-08-15

Class to read Tobii data (exported with Tobii Studio V3 and higher). See sample data in the "sampledata" folder.

Authors: Mike Wu (creator), Sebastien Lalle.
Institution: The University of British Columbia.
"""

from EMDAT_core.Recording import Recording
from EMDAT_core.data_structures import Datapoint, Fixation, Saccade, Event
import EMDAT_core.utils
import csv
import params


class TobiiV3Recording(Recording):
    def read_all_data(self, all_file):
        """Returns a list of "Datapoint"s read from an data file.

        Args:
            all_file:A string containing the name of the data file output by the Tobii software.

        Returns:
            a list of "Datapoint"s
        """
        all_data = []
        with open(all_file, 'r') as f:
            reader = csv.DictReader(f, delimiter="\t")
            last_pupil_left = -1
            last_pupil_right = -1
            last_time = -1

            for row in reader:
                for key, value in row.items():
                    row[key] = value.replace(",", ".")
            
                if row["Presented Stimulus name"] != 'Screen Recording':
#                if row["Presented Stimulus name"] != 'Screen Recordings (1)':  # ignore non-recording data point
                    continue
                if not row["Validity left"] or not row["Validity right"]: #ignore data point with no validity information
                    continue
                gaze_point_x = EMDAT_core.utils.cast_float(row["Gaze point X"], -1)
                gaze_point_y = EMDAT_core.utils.cast_float(row["Gaze point Y"], -1)
                pupil_left = EMDAT_core.utils.cast_float(row["Pupil diameter left"], -1)
                pupil_right = EMDAT_core.utils.cast_float(row["Pupil diameter right"], -1)
                distance_left = EMDAT_core.utils.cast_float(row["Eye position left Z (DACSmm)"], -1)
                distance_right = EMDAT_core.utils.cast_float(row["Eye position right Z (DACSmm)"], -1)
                timestamp = EMDAT_core.utils.cast_int(row["Recording timestamp"])
                try:
                    is_valid = True if EMDAT_core.utils.cast_int(row["Validity right"]) < 2 or EMDAT_core.utils.cast_int(row["Validity left"]) < 2 else False
                    is_valid_blink = True if EMDAT_core.utils.cast_int(row["Validity right"]) < 2 and EMDAT_core.utils.cast_int(row["Validity left"]) < 2 else False
                except:
                    is_valid = True if row["Validity right"] == "Valid" or row["Validity left"] == "Valid" else False
                    is_valid_blink = True if row["Validity right"] == "Valid" and row["Validity left"] == "Valid" else False
                if not row["Gaze point X"] or not row["Gaze point Y"]:
                    is_valid = False
                    is_valid_blink = False
                
                data = {'participant_name': row["Participant name"],
                        "timestamp": timestamp,
                        "pupilsize": EMDAT_core.Recording.get_pupil_size(pupil_left, pupil_right),
                        "pupilvelocity": EMDAT_core.Recording.get_pupil_velocity(last_pupil_left, last_pupil_right, pupil_left, pupil_right, (timestamp-last_time) ),
                        "distance": EMDAT_core.Recording.get_distance(distance_left, distance_right),
                        "is_valid": is_valid,
                        "is_valid_blink": is_valid_blink,
                        "stimuliname": row["Presented Stimulus name"],
                        "Eye movement type index": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Eye movement type index"])),
                        "gazepointx": gaze_point_x,
                        "gazepointy": gaze_point_y}
                all_data.append(Datapoint(data))
                last_pupil_left = pupil_left
                last_pupil_right = pupil_right
                last_time = timestamp

        return all_data

    def read_fixation_data(self, fixation_file):
        """Returns a list of "Fixation"s read from the data file file.

        Args:
            fixation_file: A string containing the name of the data file output by the Tobii software.

        Returns:
            a list of "Fixation"s
        """

        all_fixation = []
        with open(fixation_file, 'r') as f:
            currentfix = 0
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                for key, value in row.items():
                    row[key] = value.replace(",", ".")
                if row["Presented Stimulus name"] != 'Screen Recording' or not row["Eye movement type index"]:
#                if row["Presented Stimulus name"] != 'Screen Recordings (1)':  # ignore non-recording data point
                    continue
                if not row["Validity left"] or not row["Validity right"] or not row["Fixation point X"] or not row["Fixation point Y"] or not row["Gaze point X"]: #ignore data point with no information
                    continue
                if row["Eye movement type"] != "Fixation" or currentfix == row["Eye movement type index"]: #if not a fixation or the current fixation
                    continue
                if not row["Gaze event duration"] or not row["Recording timestamp"] or not row["Fixation point X"] or not row["Fixation point Y"]: #if not a fixation or the current fixation
                    continue
                data = {"Eye movement type index": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Eye movement type index"])),
                        "timestamp": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Recording timestamp"])),
                        "fixationduration": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze event duration"])),
                        "fixationpointx": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Fixation point X"])),
                        "fixationpointy": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Fixation point Y"]))}
                if not data["Eye movement type index"] or not data["fixationpointx"]:
                    print(row)
                    print (data)
                all_fixation.append(Fixation(data, self.media_offset))
                currentfix = row["Eye movement type index"]

        return all_fixation

    def read_saccade_data(self, saccade_file):
        """Returns a list of "Saccade"s read from the data file file.

        Args:
            fixation_file: A string containing the name of the data file output by the Tobii software.

        Returns:
            a list of "Saccade"s
        """

        all_saccade = []
        with open(saccade_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            in_saccade = False
            in_fixation = False
            last_gaze_coord = (0, 0, 0) #timestamp X Y
            saccade_vect = []
            saccade_duration = 0
            current_index = 0

            nb_invalid_temp = 0
            nb_valid_sample = 0
            nb_sample = 0

            for row in reader:
                for key, value in row.items():
                    row[key] = value.replace(",", ".")
                if row["Presented Stimulus name"] != 'Screen Recording' or not row["Recording timestamp"]:
#                if row["Presented Stimulus name"] != 'Screen Recordings (1)' or not row["Recording timestamp"]:  # ignore non-recording data point
                    continue
                    
                try:
                    is_valid = True if EMDAT_core.utils.cast_int(row["Validity right"]) < 2 or EMDAT_core.utils.cast_int(row["Validity left"]) < 2 else False
                except:
                    is_valid = True if row["Validity right"] == "Valid" or row["Validity left"] == "Valid" else False

                if in_fixation:
                    if row["Eye movement type"] == "Fixation":
                        nb_invalid_temp = 0
                    elif row["Eye movement type"] == "Saccade": #new saccade
                        in_fixation = False
                        in_saccade = True
                        current_index = row["Eye movement type index"]
                        saccade_vect = [last_gaze_coord]
                        nb_valid_sample = 0

                        #add current sample
                        if is_valid and row["Gaze point X"] and row["Gaze point Y"]: #ignore data point with no valid data
                            saccade_vect.append( [EMDAT_core.utils.cast_int(row["Recording timestamp"]), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point X"])), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point Y"]))] )
                            nb_valid_sample += 1

                        if last_valid:
                            nb_valid_sample += 1

                        nb_sample = 2 + nb_invalid_temp #current gaze sample + last gaze sample of the previous fixation + eventually all unclasified gaze samples in between
                        nb_invalid_temp = 0
                    else: #unclassified gaze samples
                        nb_invalid_temp += 1

                elif in_saccade:
                    if row["Eye movement type"] == "Fixation":
                        in_fixation = True
                        in_saccade = False

                        #end of last saccade
                        if is_valid and row["Gaze point X"] and row["Gaze point Y"]: #valid last datapoint
                            saccade_vect.append( [EMDAT_core.utils.cast_int(row["Recording timestamp"]), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point X"])), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point Y"]))] )
                            nb_valid_sample += 1
                        elif (row["Fixation point X"] and row["Fixation point Y"]): #if gaze sample not valid, try to use fixation data instead
                            saccade_vect.append( [EMDAT_core.utils.cast_int(row["Recording timestamp"]), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Fixation point X"])), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Fixation point Y"]))] )
                            nb_valid_sample += 1
                        nb_sample += 1

                        rate_valid_sample = float(nb_valid_sample) / nb_sample
                        if rate_valid_sample >= params.VALID_SAMPLES_PROP_SACCADE: #if saccade quality is above the threshold
                            saccade_duration = EMDAT_core.utils.cast_int(row["Recording timestamp"]) - saccade_vect[0][0]
                            dist = EMDAT_core.Recording.get_saccade_distance(saccade_vect)
                            accel = -1#Recording.get_saccade_acceleration(saccade_vect)
                            speed = float(dist) / EMDAT_core.utils.cast_int(saccade_duration)
                            data = {"Eye movement type index": EMDAT_core.utils.cast_int(current_index),
                                    "timestamp": saccade_vect[0][0],
                                    "saccadeduration": EMDAT_core.utils.cast_int(saccade_duration),
                                    "saccadestartpointx": saccade_vect[0][1],
                                    "saccadestartpointy": saccade_vect[0][2],
                                    "saccadeendpointx": saccade_vect[-1][1],
                                    "saccadeendpointy": saccade_vect[-1][2],
                                    "saccadedistance": dist,
                                    "saccadespeed": speed,
                                    "saccadeacceleration": accel,
									"saccadequality": rate_valid_sample
                                    }
                            all_saccade.append(Saccade(data, self.media_offset))
                            nb_valid_sample = 0
                            nb_sample = 0

                    elif row["Eye movement type"] == "Saccade":
                        if is_valid and row["Gaze point X"] and row["Gaze point Y"]: #ignore data point with no valid data
                            saccade_vect.append( [EMDAT_core.utils.cast_int(row["Recording timestamp"]), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point X"])), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point Y"]))] )
                            nb_valid_sample += 1
                        nb_sample += 1
                    else: #unclassified gaze samples
                        nb_sample += 1
                    nb_invalid_temp = 0

                else: #wait for the first fixation
                    if row["Eye movement type"] == "Fixation":
                        in_fixation = True

                if row["Gaze point X"] and row["Gaze point Y"]:
                    last_gaze_coord = (EMDAT_core.utils.cast_int(row["Recording timestamp"]), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point X"])), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Gaze point Y"])))
                    last_valid = is_valid
                elif row["Eye movement type"] == "Fixation" and row["Fixation point X"] and row["Fixation point Y"]: #if last sample not valid, at least check if valid data about the fixation
                    last_gaze_coord = (EMDAT_core.utils.cast_int(row["Recording timestamp"]), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Fixation point X"])), EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Fixation point Y"])))
                    last_valid = True

        return all_saccade


    def read_event_data(self, event_file):
        """Returns a list of "Event"s read from an data file.

        Args:
            event_file: A string containing the name of the data file output by the Tobii software.

        Returns:
            a list of "Event"s
        """

        all_event = []
        with open(event_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                for key, value in row.items():
                    row[key] = value.replace(",", ".")
                if row["Presented Stimulus name"] != 'Screen Recording':
                #if row["Presented Stimulus name"] != 'Screen Recordings (1)':  # ignore non-recording data point
                    continue
                if row["Event"] : #mouse event
                    data = {"timestamp": EMDAT_core.utils.cast_int(row["Recording timestamp"]),
                        "event": row["MouseEvent"]+"MouseClick",
                        "x_coord": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Mouse position X"])),
                        "y_coord": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Mouse position Y"]))
                        }
                    all_event.append(Event(data, self.media_offset))
                elif row["Event"] : #keyboard event
                    data = {"timestamp": EMDAT_core.utils.cast_int(EMDAT_core.utils.cast_float(row["Recording timestamp"])),
                        "event": "KeyPress",
                        "key_name": row["Event value"]
                        }
                    all_event.append(Event(data, self.media_offset))

        return all_event
