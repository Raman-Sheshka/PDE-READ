import numpy as np;
import pandas as pd;
from matplotlib import pyplot as plt;
import os
import h5py

from Create_Data_Set import Create_Data_Set;



Make_Plot : bool = True;
CURRENT_PATH = os.getcwd()

def main():
    # Specify settings.
    Data_File_Name      : str   = "Tr_G_WT_normalized_focused";
    Noise_Proportion    : float = 0.1;

    Num_Train_Examples  : int   = 5000;
    Num_Test_Examples   : int   = 1000;

    average_window = 5;
    average_min_periods = 1;
    rolling_average_type = 'gaussian';
    sigma_space = 4;
    nb_x_points = 252;
    nb_t_points = 242;
    time_born_change : bool = True;
    time_frame_start : int = 30;
    time_frame_end : int = 128;

    normalize_signal : bool = True;

    # Now pass them to "From_MATLAB".
    From_Morphogenesis(    Data_File_Name      = Data_File_Name,
                           Noise_Proportion    = Noise_Proportion,
                           Num_Train_Examples  = Num_Train_Examples,
                           Num_Test_Examples   = Num_Test_Examples,
                           average_window = average_window,
                           average_min_periods = average_min_periods,
                           rolling_average_type = rolling_average_type,
                           sigma_space = sigma_space,
                           nb_x_points = nb_x_points,
                           nb_t_points = nb_t_points,
                           time_born_change = time_born_change,
                           time_frame_start = time_frame_start,
                           time_frame_end = time_frame_end,
                           normalize_signal = normalize_signal);



def From_Morphogenesis(    Data_File_Name      : str,
                           Noise_Proportion    : float,
                           Num_Train_Examples  : int,
                           Num_Test_Examples   : int,
                           average_window      : int,
                           average_min_periods : int,
                           rolling_average_type: str,
                           sigma_space         : float,
                           nb_x_points         : int,
                           nb_t_points         : int,
                           time_born_change    : bool,
                           time_frame_start    :int,
                           time_frame_end      :int,
                           normalize_signal    :bool) -> None:
    """ This function loads a .mat data set, and generates a sparse and noisy
    data set from it. To do this, we first read in a .mat data set. We assume
    this file  contains three fields: t, x, and usol. t and x are ordered lists
    of the x and t grid lines (lines along which there are gridpoints),
    respectively. We assume that the values in x are uniformly spaced. u sol
    contains the value of the true solution at each gridpoint. Each row of usol
    contains the solution for a particular position, while each column contains
    the solution for a particular time.

    We then add the desired noise level (Noise_Proportion*100% noise) to usol,
    yielding a noisy data set. Next, we draw a sample of Num_Train_Examples
    from the set of coordinates, along with the corresponding elements of noisy
    data set. This becomes our Training data set. We draw another sample of
    Num_Test_Examples from the set of coordinates along with the
    corresponding elements of the noisy data set. These become our Testing set.

    Note: This function is currently hardcoded to work with data involving 1
    spatial dimension.

    ----------------------------------------------------------------------------
    Arguments:

    Data_File_Name: A string containing the name of a .mat file (without the
    extension) that houses the matlab data set we want to read.

    Noise_Proportion: The noise level we want to introduce.

    Num_Train_Examples, Num_Test_Examples: The number of Training/Testing
    examples we want, respectively.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Load data file.
    Data_Set_Signal_Path  = "../morphogenesis/dataset_experiment/signal_matrix_space_data.csv";
    signal_data = pd.read_csv(Data_Set_Signal_Path);
    Data_Set_Time_Path  = "../morphogenesis/dataset_experiment/plot_time_label_df.csv";
    time_data = pd.read_csv(Data_Set_Time_Path);
    space_coordinate = np.squeeze(signal_data['nx'].values);
    print('Shape of grid space_coordinate:', space_coordinate.shape);
    time_coordinate = time_data['frame_in_hours'].values;
    u_data = signal_data.copy();
    u_data = u_data.drop(columns=['nx']);
   
    if time_born_change: 
        time_coordinate = time_coordinate[time_frame_start:time_frame_end];
        u_data = u_data.iloc[:,time_frame_start:time_frame_end].copy();
    print('Shape of grid u_data:', u_data.shape);
    time_coordinate_offset = time_coordinate.min();
    time_coordinate = time_coordinate - time_coordinate_offset;
    space_coordinate_offset = space_coordinate.min();
    space_coordinate = space_coordinate - space_coordinate_offset;
    print('Shape of grid time_coordinate:', time_coordinate.shape);
    #resize the input data to 'nb_x_points'x'nb_t_points'
    average_window_characteristics = [average_window, 
                                      average_min_periods,
                                      rolling_average_type,
                                      sigma_space];
    u_data_copy = u_data.copy();
    u_data_copy.columns = range(u_data_copy.shape[1]);
    u_data_copy.reset_index(drop=True, inplace=True);

    x_new, t_new, u_new = signal_rebase(u_data_copy,
                                        space_coordinate,
                                        time_coordinate,
                                        nb_x_points,
                                        nb_t_points,
                                        average_window_characteristics);
    if normalize_signal:
        u_new_min = u_new.min().min();
        u_new_max = u_new.max().max();
        u_new_normalized = (u_new - u_new_min)/(u_new_max - u_new_min)*2-1;
        u_new = u_new_normalized;
    t,x = np.meshgrid(t_new,x_new);
    print('Shape of grid u_data resized:',  u_new.shape);
    data_in = {};
    data_in['t'] = t;
    print('Shape of grid t:', data_in['t'].shape);
    data_in['x'] = x;
    print('Shape of grid x:', data_in['x'].shape);
    data_in['u'] = u_new;
    print('Shape of grid u:', data_in['u'].shape);

    # Fetch spatial, temporal coordinates and the true solution. We cast these
    # to singles (32 bit fp) since that's what PDE-REAd uses.
    t_points    =  t_new.reshape(-1).astype(dtype = np.float32);
    print('number t points:',t_points.shape[0])
    x_points    =  x_new.reshape(-1).astype(dtype = np.float32);
    print('number x points:',x_points.shape[0])
    Data_Set    = (np.real(u_new)).astype( dtype = np.float32);
    print('number points Data_Set:',Data_Set.shape)

    # Determine problem bounds.
    Input_Bounds : np.ndarray    = np.empty(shape = (2, 2), dtype = np.float32);
    Input_Bounds[0, 0]              = t_points[ 0];
    Input_Bounds[0, 1]              = t_points[-1];
    Input_Bounds[1, 0]              = x_points[ 0];
    Input_Bounds[1, 1]              = x_points[-1];

    # Add noise to true solution.
    Noisy_Data_Set = Data_Set + (Noise_Proportion)*np.std(Data_Set)*np.random.randn(*Data_Set.shape);

    # Generate the grid of (t, x) coordinates where we'll enforce the "true
    # solution". Each row of these arrays corresponds to a particular position.
    # Each column corresponds to a particular time.
    t_coords_matrix, x_coords_matrix  = np.meshgrid(t_points, x_points);
    print('number points Noisy_Data_Set:', Noisy_Data_Set.shape)

    if(Make_Plot == True):
        epsilon : float = .0001;
        Data_min : float = np.min(Noisy_Data_Set) - epsilon;
        Data_max : float = np.max(Noisy_Data_Set) + epsilon;

        # Plot!
        plt.contourf(    t_coords_matrix,
                            x_coords_matrix,
                            Noisy_Data_Set,
                            levels      = np.linspace(Data_min, Data_max, 500),
                            cmap        = plt.cm.jet);

        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.show();

        Data_min : float = np.min(Data_Set);
        Data_max : float = np.max(Data_Set);
        plt.contourf(       t_coords_matrix,
                            x_coords_matrix,
                            Data_Set,
                            levels      = np.linspace(Data_min, Data_max, 500),
                            cmap        = plt.cm.jet);

        plt.colorbar();
        plt.xlabel("t");
        plt.ylabel("x");
        plt.show();

    # Now, stitch successive the rows of the coordinate matrices together
    # to make a 1D array. We interpert the result as a 1 column matrix.
    t_coords_1D : np.ndarray = t_coords_matrix.flatten().reshape(-1, 1);
    x_coords_1D : np.ndarray = x_coords_matrix.flatten().reshape(-1, 1);

    # Generate data coordinates, corresponding Data Values.
    All_Data_Coords : np.ndarray = np.hstack((t_coords_1D, x_coords_1D));
    All_Data_Values : np.ndarray = Noisy_Data_Set.flatten();

    # Next, generate the Testing/Training sets. To do this, we sample a uniform
    # distribution over subsets of {1, ... , N} of size Num_Train_Examples,
    # and another over subsets of {1, ... , N} of size Num_Test_Examples.
    # Here, N is the number of coordinates.
    Train_Indicies : np.ndarray = np.random.choice(All_Data_Coords.shape[0], Num_Train_Examples, replace = False);
    Test_Indicies  : np.ndarray = np.random.choice(All_Data_Coords.shape[0], Num_Test_Examples , replace = False);

    # Now select the corresponding testing, training data points/values.
    Train_Inputs    = All_Data_Coords[Train_Indicies, :];
    Train_Targets   = All_Data_Values[Train_Indicies];

    Test_Inputs     = All_Data_Coords[Test_Indicies, :];
    Test_Targets    = All_Data_Values[Test_Indicies];

    # Send everything to Create_Data_Set
    DataSet_Name : str = (  Data_File_Name + "_" +
                            "N" + str(int(100*Noise_Proportion)) + "_" +
                            "P" + str(Num_Train_Examples) );

    Create_Data_Set(    Name            = DataSet_Name,
                        Train_Inputs    = Train_Inputs,
                        Train_Targets   = Train_Targets,
                        Test_Inputs     = Test_Inputs,
                        Test_Targets    = Test_Targets,
                        Input_Bounds    = Input_Bounds);

    save_rescaled_data_file_name = DataSet_Name + ".hdf5";
    if os.path.isfile(os.path.join(CURRENT_PATH,"DataSets",save_rescaled_data_file_name)):
        os.remove(os.path.join(CURRENT_PATH,"DataSets",save_rescaled_data_file_name));
    os.chdir(os.path.join(CURRENT_PATH,"DataSets"));                                   
    outfile = h5py.File(save_rescaled_data_file_name , 'w');
    outfile.close();
    outfile =  h5py.File(save_rescaled_data_file_name , 'w');
    outfile.create_dataset('t', data = t_new);
    outfile.create_dataset('x', data = x_new);
    outfile.create_dataset('u', data = u_new);
    outfile.close();
    os.chdir(CURRENT_PATH);                    

def signal_rebase(matrix : pd.DataFrame(), x_ref : np.array(()), t_ref : np.array(()), nb_x : int, nb_t : int, average_window_characteristics):
    average_window = average_window_characteristics[0];
    average_min_periods = 1;
    rolling_average_type = 'gaussian';
    sigma_space = average_window_characteristics[3];
    matrix_interpolated_temporal = matrix.copy();
    matrix_interpolated_temporal.reset_index(drop=True, inplace=True);
    x_min = x_ref.min();
    x_max = x_ref.max();
    print('x_min:',x_min);
    print('x_max:',x_max);
    t_min = t_ref.min();
    t_max = t_ref.max();
    print('t_min:',x_min);
    print('t_max:',x_max);
    x_new = np.linspace(x_min, x_max, nb_x);
    t_new = np.linspace(t_min, t_max, nb_t);
    matrix_temporal = [];
    for column in range(matrix_interpolated_temporal.columns.size):
        ny = np.interp(x_new, x_ref, matrix_interpolated_temporal[column]);
        matrix_temporal.append(ny);
    matrix_temporal_df = pd.DataFrame(matrix_temporal);
    matrix_temporal_df.rolling(window = average_window,
                               min_periods = average_min_periods,
                               win_type = rolling_average_type).mean(std = sigma_space);
    matrix_temporal = [];
    for column in range(matrix_temporal_df.columns.size):
        ny = np.interp(t_new, t_ref, matrix_temporal_df[column]);
        matrix_temporal.append(ny);
    matrix_new = pd.DataFrame(matrix_temporal);
    matrix_new.rolling(window = average_window,
                       min_periods = average_min_periods,
                       win_type = rolling_average_type).mean(std = sigma_space);
    return x_new, t_new, matrix_new 

if __name__ == "__main__":
    main();
