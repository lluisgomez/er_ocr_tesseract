import subprocess
import sys

alt_name = "Feedback Loop"

with open("test/list.txt") as f:
  content = f.readlines()

total_edit_distance = 0
edit_distance_ratio = 0.0
total_edit_distance_alt = 0
edit_distance_ratio_alt = 0.0
total_time_regions = 0.0
total_time_regions_alt = 0.0
total_time_grouping = 0.0
total_time_grouping_alt = 0.0
total_time_ocr = 0.0
total_time_ocr_alt = 0.0
counter = 0
tp=0
fp=0
fn=0
tp_alt=0
fp_alt=0
fn_alt=0
for line in content:
  print line
  if (counter>=0):
    args  = line.split()
    ident = args[0].split("/")
    ident = ident[1].split(".")
    ident = ident[0]
    with open('tmp.txt', "w") as outfile:
      outfile.write('ID=\''+ident+'\'\n')
    with open('tmp.txt', "a") as outfile:
      subprocess.call(['./end_to_end_recognition']+line.split(),stdout=outfile)
      subprocess.call(['./pipeline_comparison']+line.split(),stdout=outfile)

    execfile('tmp.txt')
    total_edit_distance += TOTAL_EDIT_DISTANCE
    edit_distance_ratio += EDIT_DISTANCE_RATIO
    total_edit_distance_alt += TOTAL_EDIT_DISTANCE_ALT
    edit_distance_ratio_alt += EDIT_DISTANCE_RATIO_ALT
    total_time_regions += TIME_REGION_DETECTION
    total_time_regions_alt += TIME_REGION_DETECTION_ALT
    total_time_grouping += TIME_GROUPING
    total_time_grouping_alt += TIME_GROUPING_ALT
    total_time_ocr += TIME_OCR
    total_time_ocr_alt += TIME_OCR_ALT
    tp += TP 
    tp_alt += TP_ALT
    fp += FP 
    fp_alt += FP_ALT
    fn += FN 
    fn_alt += FN_ALT

    #convert label on top of original image
    #subprocess.call(["convert", args[0], "-geometry", "640x", "tmp1.jpg"])
    subprocess.call(["convert", "decomposition.jpg", "-geometry", "640x", "tmp1.jpg"])
    label = "Image size "+str(IMG_W)+"x"+str(IMG_H)+" pixels"
    subprocess.call(["convert", "tmp1.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp1.jpg"])
    label = "Region detection (2 channels) = "+str(int(TIME_REGION_DETECTION))+" ms."
    subprocess.call(["convert", "tmp1.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp1.jpg"])
    label = " "
    subprocess.call(["convert", "tmp1.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp1.jpg"])
    #convert label on top of detection image
    subprocess.call(["convert", "detection.jpg", "-geometry", "640x", "tmp2.jpg"])
    label = " "
    subprocess.call(["convert", "tmp2.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp2.jpg"])
    label = "Grouping         (2 channels) = "+str(int(TIME_GROUPING))+" ms."
    subprocess.call(["convert", "tmp2.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp2.jpg"])
    label = " "
    subprocess.call(["convert", "tmp2.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp2.jpg"])
    #convert label on top of segmentation image
    subprocess.call(["convert", "segmentation.jpg", "-geometry", "640x", "tmp3.jpg"])
    label = " "
    subprocess.call(["convert", "tmp3.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp3.jpg"])
    label = "Segmentation (what we send to the OCR)"
    subprocess.call(["convert", "tmp3.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp3.jpg"])
    label = " "
    subprocess.call(["convert", "tmp3.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp3.jpg"])
    #convert label on top of recognition image
    subprocess.call(["convert", "recognition.jpg", "-geometry", "640x", "tmp4.jpg"])
    label = "OCR Recognition  (all groups) = "+str(int(TIME_OCR))+" ms."
    subprocess.call(["convert", "tmp4.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp4.jpg"])
    label = "OCR initialization        = "+str(int(TIME_OCR_INITIALIZATION))+" ms."
    subprocess.call(["convert", "tmp4.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp4.jpg"])
    label = "Edist distance ratio      = "+str(EDIT_DISTANCE_RATIO)
    subprocess.call(["convert", "tmp4.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp4.jpg"])
    # montage of the top row
    subprocess.call(["montage", "tmp1.jpg", "tmp2.jpg", "tmp3.jpg", "tmp4.jpg", "-tile", "4x1","-geometry","640x+10+10","tmp_montage1.jpg"])
    subprocess.call(["convert","tmp_montage1.jpg","-rotate","90", "-background", "white", "-size", "x31", "label:BASELINE","+swap","-gravity","Center","-append","-rotate","-90 ","tmp_montage1.jpg"])
    subprocess.call(["convert","tmp_montage1.jpg","-rotate","90", "-background", "white", "-size", "x31", "label: ","-gravity","Center","-append","-rotate","-90 ","tmp_montage1.jpg"])


    #convert label on top of original image alternative
    #subprocess.call(["convert", args[0], "-geometry", "640x", "tmp5.jpg"])
    subprocess.call(["convert", "decomposition_alt.jpg", "-geometry", "640x", "tmp5.jpg"])
    label = "Image size "+str(IMG_W)+"x"+str(IMG_H)+" pixels"
    subprocess.call(["convert", "tmp5.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp5.jpg"])
    label = "Region detection (2 channels) = "+str(int(TIME_REGION_DETECTION_ALT))+" ms."
    subprocess.call(["convert", "tmp5.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp5.jpg"])
    label = " "
    subprocess.call(["convert", "tmp5.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp5.jpg"])
    #convert label on top of detection image alternative
    subprocess.call(["convert", "detection_alt.jpg", "-geometry", "640x", "tmp6.jpg"])
    label = " "
    subprocess.call(["convert", "tmp6.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp6.jpg"])
    label = "Grouping         (2 channels) = "+str(int(TIME_GROUPING_ALT))+" ms."
    subprocess.call(["convert", "tmp6.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp6.jpg"])
    label = " "
    subprocess.call(["convert", "tmp6.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp6.jpg"])
    #convert label on top of segmentation image alternative
    subprocess.call(["convert", "segmentation_alt.jpg", "-geometry", "640x", "tmp7.jpg"])
    label = " "
    subprocess.call(["convert", "tmp7.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp7.jpg"])
    label = "Segmentation (what we send to the OCR)"
    subprocess.call(["convert", "tmp7.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp7.jpg"])
    label = " "
    subprocess.call(["convert", "tmp7.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp7.jpg"])
    #convert label on top of recognition image alternative
    subprocess.call(["convert", "recognition_alt.jpg", "-geometry", "640x", "tmp8.jpg"])
    label = "OCR Recognition  (all groups) = "+str(int(TIME_OCR_ALT))+" ms."
    subprocess.call(["convert", "tmp8.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp8.jpg"])
    label = "OCR initialization        = "+str(int(TIME_OCR_INITIALIZATION_ALT))+" ms."
    subprocess.call(["convert", "tmp8.jpg", "-background", "white", "-size", "x31", "label:"+label, "+swap", "-gravity", "Center", "-append", "tmp8.jpg"])
    label = "Edist distance ratio      = "+str(EDIT_DISTANCE_RATIO_ALT)
    subprocess.call(["convert", "tmp8.jpg", "-background", "white", "-size", "x31", "label:"+label, "-gravity", "Center", "-append", "tmp8.jpg"])
    # montage of the bottom row
    subprocess.call(["montage", "tmp5.jpg", "tmp6.jpg", "tmp7.jpg", "tmp8.jpg", "-tile", "4x1","-geometry","640x+10+10","tmp_montage2.jpg"])
    subprocess.call(["convert","tmp_montage2.jpg","-rotate","90", "-background", "white", "-size", "x31", "label:"+alt_name,"+swap","-gravity","Center","-append","-rotate","-90 ","tmp_montage2.jpg"])
    subprocess.call(["convert","tmp_montage2.jpg","-rotate","90", "-background", "white", "-size", "x31", "label: ","-gravity","Center","-append","-rotate","-90 ","tmp_montage2.jpg"])


    # montage of the comparison page
    name_sort = "{:.4f}".format(EDIT_DISTANCE_RATIO)
    subprocess.call(["montage", "tmp_montage1.jpg", "tmp_montage2.jpg", "-tile", "1x2","-geometry","2702x+1+13","results/"+str(name_sort)+"-page-"+str(ID)+".jpg"])


    counter = counter+1

print "Total edit distance          = "+str(total_edit_distance)
print "Avg. edit distance ratio     = "+str(edit_distance_ratio/counter)
print "Avg. time regions extraction = "+str(total_time_regions/counter)
print "Avg. time grouping           = "+str(total_time_grouping/counter)
print "Avg. time ocr                = "+str(total_time_ocr/counter)
print "End-to-end F-score           = "+str(2.0*tp/(2*tp+fp+fn)) 
print "(alternative) Total edit distance          = "+str(total_edit_distance_alt)
print "(alternative) Avg. edit distance ratio     = "+str(edit_distance_ratio_alt/counter)
print "(alternative) Avg. time regions extraction = "+str(total_time_regions_alt/counter)
print "(alternative) Avg. time grouping           = "+str(total_time_grouping_alt/counter)
print "(alternative) Avg. time ocr                = "+str(total_time_ocr_alt/counter)
print "(alternative) End-to-end F-score           = "+str(2.0*tp_alt/(2*tp_alt+fp_alt+fn_alt)) 
quit()

