import numpy;

import os;
from Settings_Reader import Settings_Reader;
from utils           import create_record_history_logfile, create_record_pde_extraxted, copy_and_rename, create_folder_if_not_exists;

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    ############################################################################
    # Load settings, print them.
    Settings = Settings_Reader();
    if os.path.isfile('../' + 'Settings.txt'):
            print('yes')
            filename: str = 'Settings.txt';
            copy_and_rename('../Settings.txt',
                            "../Saves/" + Settings.Save_Folder_Name + "/" + filename);
    else:
        print('no')        

if __name__ == '__main__':
    main();    
     