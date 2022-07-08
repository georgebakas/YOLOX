import os
import glob 

# Absolute path of a file
old_name = r""
new_name = r""
directories = ['test', 'training', 'validation']

for dir_ in directories:
    for file in glob.iglob(f'/home/studio-lab-user/YOLOX/datasets/images/{dir_}/*'):
        print(file)
        new_name = file.replace('(','')
        new_name = new_name.replace(')','')
        print('new: ', new_name)
        os.rename(file, new_name)
        #break
    #break
                           