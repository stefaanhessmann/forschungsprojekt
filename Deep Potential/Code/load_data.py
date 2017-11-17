import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

path_to_files = "./../Dataset/dsC7O2H10nsd.xyz/dsC7O2H10nsd_0001.xyz"
df_input = pd.read_csv(path_to_files, names=['atom', 'x', 'y', 'z', 'unknown'], delimiter='\t', skiprows=2, nrows=19)
df_input['r'] = list(zip(df_input.x, df_input.y, df_input.z))
#df_input.drop(['x', 'y', 'z'], axis=1)

distance_matrix =  pd.DataFrame(distance_matrix(df_input.loc(['x', 'y', 'z']), df_input.loc(['x', 'y', 'z'])))#, index=df_input.index, columns=df_input.index)
