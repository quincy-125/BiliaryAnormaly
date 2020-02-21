import os

# create bash file for batch processing
data_dir = "//mfad.mfroot.org/researchmn/DLMP-MACHINE-LEARNING/BiliaryCytology/req19175_1-22-2019"
data_out_dir = "H:\\BiliaryStains\\WSI_extraction"
wsi_names = os.listdir(data_dir)
out_str = ""
for fn in wsi_names:
    WSI_file = data_dir + "/" + fn
    out_str += ("python WSI_extraction_sh.py -n \"%s\"\n" %fn)

with open("extract_all.bat", "w") as fp:
    fp.write(out_str)
















