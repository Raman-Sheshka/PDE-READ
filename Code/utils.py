import os
import csv

def create_record_history_logfile(file_path, data_to_write):
    header_row = ['epoch',
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