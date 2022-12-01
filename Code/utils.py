import os
import csv

def create_record_history_logfile(file_path, data_to_write)->None:
    header_row = ['epoch',
                  'optimizer', 
                  'experiment',
                  'collocation',
                  'data',
                  'total'
                  ]
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data_to_write)
            # close the file
            f.close()
    else:
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            # write a header row to the csv file
            writer.writerow(header_row)
            # write a data row to the csv file
            writer.writerow(data_to_write)
            # close
            f.close()

def create_record_pde_extraxted(file_path, data_to_write)->None:
    header_row = ['nb',
                  'rank', 
                  'residual_ranked',
                  'residual_change',
                  'extracted_pde'
                  ]       
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data_to_write)
            # close the file
            f.close()
    else:
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            # write a header row to the csv file
            writer.writerow(header_row)
            # write a data row to the csv file
            writer.writerow(data_to_write)
            # close
            f.close()