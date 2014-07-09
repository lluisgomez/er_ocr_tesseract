import subprocess
import random
import string

msfontdir = "/usr/share/fonts/truetype/msttcorefonts/"

#generate some synthetic characters from MS core fonts
fonts = ["Andale_Mono.ttf","andalemo.ttf","arialbd.ttf","arialbi.ttf","Arial_Black.ttf","Arial_Bold_Italic.ttf","Arial_Bold.ttf","Arial_Italic.ttf","ariali.ttf","arial.ttf","Arial.ttf","ariblk.ttf","comicbd.ttf","Comic_Sans_MS_Bold.ttf","Comic_Sans_MS.ttf","comic.ttf","courbd.ttf","courbi.ttf","Courier_New_Bold_Italic.ttf","Courier_New_Bold.ttf","Courier_New_Italic.ttf","Courier_New.ttf","couri.ttf","cour.ttf","Georgia_Bold_Italic.ttf","Georgia_Bold.ttf","georgiab.ttf","Georgia_Italic.ttf","georgiai.ttf","georgia.ttf","Georgia.ttf","georgiaz.ttf","impact.ttf","Impact.ttf","timesbd.ttf","timesbi.ttf","timesi.ttf","Times_New_Roman_Bold_Italic.ttf","Times_New_Roman_Bold.ttf","Times_New_Roman_Italic.ttf","Times_New_Roman.ttf","times.ttf","trebucbd.ttf","trebucbi.ttf","Trebuchet_MS_Bold_Italic.ttf","Trebuchet_MS_Bold.ttf","Trebuchet_MS_Italic.ttf","Trebuchet_MS.ttf","trebucit.ttf","trebuc.ttf","Verdana_Bold_Italic.ttf","Verdana_Bold.ttf","verdanab.ttf","Verdana_Italic.ttf","verdanai.ttf","verdana.ttf","Verdana.ttf","verdanaz.ttf"];
#number of sample of each class
num_samples = len(fonts) 
for letter in range(0,62):
  for i in range(0,num_samples):
    #c = random.choice(string.ascii_letters);
    if (letter<52):
      c = string.ascii_letters[letter]
    else:
      c = str(letter-52)
  
    gen_command = "convert -background black -fill white -font "+msfontdir+fonts[i]+" -pointsize "+str(random.randrange(100, 198))+" label:"+c+" -page +0+0 synth.tiff"
    process = subprocess.Popen(gen_command, shell=True)
    process.wait()
    #print gen_command
    out = subprocess.check_output(["./extract_features","synth.tiff",str(letter)]);
    print out
    #out = subprocess.check_output(["eog","synth.tiff"]);
    gen_command = "convert synth.tiff -scale 25% -scale 400% synth.tiff; convert synth.tiff -colorspace gray  +dither  -colors 2  -normalize  synth_noised.tiff"
    process = subprocess.Popen(gen_command, shell=True)
    process.wait()
    out = subprocess.check_output(["./extract_features","synth_noised.tiff",str(letter)]);
    print out

