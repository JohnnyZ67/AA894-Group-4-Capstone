import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

# Load the dataset
data = pd.read_csv("reduced_data.csv")

def aggregate_play(df):
    """
    Aggregates the data for each play, pivoting the table so that each row represents a single play
    and contains the x, y, o, and position data for each player.
    """
    # Select the columns we need
    df = df[['uniquePlayId', 'playDirection', 'x', 'y', 'o', 'position', 'offenseFormation']]

    # Pivot the table using a multi-index
    df_pivot = df.pivot_table(index=['uniquePlayId', 'playDirection', 'offenseFormation'], columns='position', values=['x', 'y', 'o'])

    # Flatten the multi-level column index
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    # Reset index to make 'uniquePlayId' a regular column
    df_pivot = df_pivot.reset_index()

    return df_pivot

data = aggregate_play(data)

# Print the first few rows of the aggregated data
print(data.head())

# Configure H2O
h2o_config = {
    "nthreads": -1,  # Use all available CPU threads
    "max_mem_size": "16G",  # Adjust based on your RAM
}

# Initialize H2O with GPU support
h2o.init(**h2o_config)

# Convert pandas DataFrame to H2O Frame
h2o_df = h2o.H2OFrame(data)

# Identify target and predictor columns
y = "offenseFormation"
x = h2o_df.columns
x.remove(y)
x.remove("uniquePlayId") # Remove uniquePlayId as it is only used to group rows.
x.remove("playDirection") # Remove playDirection as it is a categorical value that might confuse the data.

# Split data into training and testing sets
train, test = h2o_df.split_frame(ratios=[0.8], seed=3245)

# Configure AutoML
aml = H2OAutoML(max_runtime_secs=10_000, # Run for 1 hour (adjust as needed)
                seed=3245,
                sort_metric = "logloss",
                nfolds = 10,
                # exclude_algos = ["DRF", "GBM", "GLM", "DeepLearning"],
                # include_algos = ["XGBoost", "StackedEnsemble"],
                )

# Train the model
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

# Evaluate the Leader Model on Test Data
perf = aml.leader.model_performance(test_data=test)
print(perf)

# Get the Best Model
best_model = aml.leader

# Save the Leader Model
model_path = h2o.save_model(model=best_model, path="./model_new", force=True)
print("Model saved to: ", model_path)

# Shutdown H2O cluster
h2o.cluster().shutdown()

r'''
    uniquePlayId playDirection offenseFormation      o_C    o_FB     o_G    o_QB    o_RB      o_T  ...       x_WR     y_C   y_FB    y_G   y_QB   y_RB     y_T   y_TE       y_WR
0  2022090800-1009         right          SHOTGUN   96.270       175.57   90.42   97.49   61.375  ...  45.006667  23.015    NaN  25.58  23.71  21.16  24.075  19.27  30.093333        
1   2022090800-101          left           I_FORM  275.550  336.22  273.97  268.28  265.65  264.245  ...  72.980000  30.355  26.33  27.67  29.56  29.56  29.425  34.57  29.940000        
2  2022090800-1030         right          SHOTGUN   72.415     NaN   85.72   92.30   87.89   74.725  ...  48.010000  22.855    NaN  25.06  23.46  26.19  23.915  18.16  26.053333        
3  2022090800-1102         right          SHOTGUN   84.845     NaN   89.49   86.68  105.62   99.675  ...  83.723333  28.890    NaN  31.26  29.51  33.43  29.930  37.98  15.606667        
4  2022090800-1126         right          SHOTGUN   94.660     NaN   80.13   89.83   94.80   97.780  ...  85.893333  28.905    NaN  31.18  29.52  32.05  29.825   7.06  26.130000        

[5 rows x 27 columns]
Checking whether there is an H2O instance running at http://localhost:54321..... not found.
Attempting to start a local H2O server...
; Java HotSpot(TM) 64-Bit Server VM (build 25.441-b07, mixed mode)
  Starting server from C:\Users\quort\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\h2o\backend\bin\h2o.jar
  Ice root: C:\Users\quort\AppData\Local\Temp\tmpy1dqweek
  JVM stdout: C:\Users\quort\AppData\Local\Temp\tmpy1dqweek\h2o_quort_started_from_python.out
  JVM stderr: C:\Users\quort\AppData\Local\Temp\tmpy1dqweek\h2o_quort_started_from_python.err
  Server is running at http://127.0.0.1:54321
Connecting to H2O server at http://127.0.0.1:54321 ... successful.
Warning: Your H2O cluster version is (4 months and 18 days) old.  There may be a newer version available.
Please download and install the latest version from: https://h2o-release.s3.amazonaws.com/h2o/latest_stable.html
--------------------------  -----------------------------
H2O_cluster_uptime:         02 secs
H2O_cluster_timezone:       America/Los_Angeles
H2O_data_parsing_timezone:  UTC
H2O_cluster_version:        3.46.0.6
H2O_cluster_version_age:    4 months and 18 days
H2O_cluster_name:           H2O_from_python_quort_iy4dy3
H2O_cluster_total_nodes:    1
H2O_cluster_free_memory:    14.20 Gb
H2O_cluster_allowed_cores:  32
H2O_cluster_status:         locked, healthy
H2O_connection_url:         http://127.0.0.1:54321
H2O_connection_proxy:       {"http": null, "https": null}
H2O_internal_security:      False
Python_version:             3.11.9 final
--------------------------  -----------------------------
Parse progress: |████████████████████████████████████████████████████████████████ (done)| 100%
AutoML progress: |                                                               |   0%
01:40:16.339: AutoML: XGBoost is not available; skipping it.

AutoML progress: |███████████████████████████████████████████████████████████████ (done)| 100%
model_id                                                  logloss    mean_per_class_error      rmse       mse
StackedEnsemble_BestOfFamily_4_AutoML_1_20250321_14016   0.36432                 0.459059  0.321588  0.103419
StackedEnsemble_AllModels_2_AutoML_1_20250321_14016      0.373977                0.464194  0.325012  0.105633
StackedEnsemble_BestOfFamily_3_AutoML_1_20250321_14016   0.375806                0.461685  0.326066  0.106319
StackedEnsemble_AllModels_1_AutoML_1_20250321_14016      0.389626                0.477025  0.334963  0.1122
StackedEnsemble_BestOfFamily_2_AutoML_1_20250321_14016   0.395494                0.482292  0.338579  0.114636
GBM_grid_1_AutoML_1_20250321_14016_model_79              0.405702                0.47983   0.330999  0.10956
GBM_grid_1_AutoML_1_20250321_14016_model_7               0.40665                 0.479686  0.3315    0.109892
GBM_grid_1_AutoML_1_20250321_14016_model_131             0.408479                0.474194  0.326769  0.106778
GBM_grid_1_AutoML_1_20250321_14016_model_105             0.411792                0.474059  0.330357  0.109136
GBM_grid_1_AutoML_1_20250321_14016_model_30              0.414013                0.473243  0.329749  0.108734
GBM_grid_1_AutoML_1_20250321_14016_model_94              0.416499                0.482477  0.33445   0.111857
GBM_grid_1_AutoML_1_20250321_14016_model_47              0.416641                0.476339  0.332864  0.110798
GBM_grid_1_AutoML_1_20250321_14016_model_123             0.418669                0.476123  0.333072  0.110937
GBM_grid_1_AutoML_1_20250321_14016_model_109             0.420143                0.477332  0.334419  0.111836
GBM_grid_1_AutoML_1_20250321_14016_model_39              0.421304                0.477934  0.334314  0.111766
GBM_grid_1_AutoML_1_20250321_14016_model_42              0.422476                0.480041  0.340751  0.116111
GBM_grid_1_AutoML_1_20250321_14016_model_63              0.423024                0.482463  0.338     0.114244
StackedEnsemble_BestOfFamily_1_AutoML_1_20250321_14016   0.42413                 0.516742  0.351498  0.123551
GBM_grid_1_AutoML_1_20250321_14016_model_167             0.424145                0.481429  0.338025  0.114261
GBM_grid_1_AutoML_1_20250321_14016_model_180             0.424875                0.480212  0.337407  0.113843
GBM_grid_1_AutoML_1_20250321_14016_model_17              0.424893                0.486819  0.338577  0.114635
GBM_grid_1_AutoML_1_20250321_14016_model_132             0.425936                0.485733  0.342841  0.11754
GBM_grid_1_AutoML_1_20250321_14016_model_6               0.428868                0.486312  0.342776  0.117495
GBM_grid_1_AutoML_1_20250321_14016_model_92              0.428891                0.477137  0.343883  0.118256
GBM_grid_1_AutoML_1_20250321_14016_model_60              0.429095                0.47806   0.336558  0.113271
GBM_grid_1_AutoML_1_20250321_14016_model_64              0.430386                0.479566  0.341614  0.1167
GBM_grid_1_AutoML_1_20250321_14016_model_15              0.432496                0.482351  0.342279  0.117155
GBM_grid_1_AutoML_1_20250321_14016_model_71              0.43261                 0.483067  0.338292  0.114441
GBM_grid_1_AutoML_1_20250321_14016_model_165             0.432667                0.483577  0.34046   0.115913
GBM_grid_1_AutoML_1_20250321_14016_model_76              0.432694                0.489347  0.342205  0.117104
GBM_grid_1_AutoML_1_20250321_14016_model_4               0.432841                0.480027  0.336199  0.11303
GBM_grid_1_AutoML_1_20250321_14016_model_126             0.433698                0.481218  0.338042  0.114272
GBM_grid_1_AutoML_1_20250321_14016_model_157             0.434606                0.486569  0.344728  0.118837
GBM_grid_1_AutoML_1_20250321_14016_model_146             0.435931                0.490837  0.342925  0.117597
GBM_grid_1_AutoML_1_20250321_14016_model_91              0.43603                 0.486243  0.333498  0.111221
GBM_4_AutoML_1_20250321_14016                            0.43642                 0.479456  0.340234  0.115759
GBM_grid_1_AutoML_1_20250321_14016_model_135             0.436438                0.488119  0.337879  0.114162
GBM_grid_1_AutoML_1_20250321_14016_model_141             0.436571                0.475066  0.3382    0.114379
GBM_grid_1_AutoML_1_20250321_14016_model_88              0.437068                0.485707  0.344729  0.118838
GBM_grid_1_AutoML_1_20250321_14016_model_70              0.437527                0.485294  0.338698  0.114717
GBM_grid_1_AutoML_1_20250321_14016_model_34              0.437746                0.494118  0.343575  0.118044
GBM_grid_1_AutoML_1_20250321_14016_model_84              0.438061                0.482417  0.340187  0.115727
GBM_grid_1_AutoML_1_20250321_14016_model_56              0.438109                0.48257   0.339096  0.114986
GBM_grid_1_AutoML_1_20250321_14016_model_28              0.43813                 0.488425  0.340377  0.115857
GBM_grid_1_AutoML_1_20250321_14016_model_50              0.438323                0.482272  0.33998   0.115586
GBM_grid_1_AutoML_1_20250321_14016_model_163             0.438394                0.491927  0.340542  0.115969
GBM_grid_1_AutoML_1_20250321_14016_model_122             0.438828                0.486313  0.339769  0.115443
GBM_grid_1_AutoML_1_20250321_14016_model_103             0.438982                0.481985  0.339034  0.114944
GBM_grid_1_AutoML_1_20250321_14016_model_82              0.439229                0.482382  0.342387  0.117229
GBM_grid_1_AutoML_1_20250321_14016_model_121             0.439564                0.488192  0.342574  0.117357
GBM_3_AutoML_1_20250321_14016                            0.440002                0.485732  0.34519   0.119156
GBM_grid_1_AutoML_1_20250321_14016_model_80              0.441408                0.484749  0.342756  0.117481
GBM_2_AutoML_1_20250321_14016                            0.442272                0.48518   0.344022  0.118351
GBM_grid_1_AutoML_1_20250321_14016_model_48              0.442566                0.490007  0.344778  0.118872
GBM_grid_1_AutoML_1_20250321_14016_model_118             0.444102                0.471272  0.341087  0.116341
GBM_grid_1_AutoML_1_20250321_14016_model_13              0.444235                0.482394  0.347205  0.120552
GBM_grid_1_AutoML_1_20250321_14016_model_21              0.444319                0.508615  0.348606  0.121526
GBM_grid_1_AutoML_1_20250321_14016_model_8               0.444849                0.486081  0.345189  0.119156
GBM_grid_1_AutoML_1_20250321_14016_model_177             0.445595                0.488785  0.343856  0.118237
GBM_grid_1_AutoML_1_20250321_14016_model_1               0.445614                0.478374  0.34273   0.117464
GBM_grid_1_AutoML_1_20250321_14016_model_31              0.446068                0.487525  0.344807  0.118892
GBM_grid_1_AutoML_1_20250321_14016_model_134             0.446387                0.480338  0.344271  0.118523
GBM_grid_1_AutoML_1_20250321_14016_model_68              0.446424                0.495423  0.347707  0.1209
GBM_grid_1_AutoML_1_20250321_14016_model_143             0.446498                0.505795  0.346871  0.120319
GBM_grid_1_AutoML_1_20250321_14016_model_124             0.447033                0.485473  0.342636  0.1174
GBM_grid_1_AutoML_1_20250321_14016_model_2               0.447327                0.489266  0.34862   0.121536
GBM_grid_1_AutoML_1_20250321_14016_model_61              0.447717                0.478637  0.34532   0.119246
GBM_grid_1_AutoML_1_20250321_14016_model_54              0.448491                0.485043  0.346071  0.119765
GBM_grid_1_AutoML_1_20250321_14016_model_35              0.44862                 0.480424  0.344819  0.1189
GBM_grid_1_AutoML_1_20250321_14016_model_153             0.448653                0.488931  0.34008   0.115654
GBM_grid_1_AutoML_1_20250321_14016_model_89              0.44958                 0.482275  0.337593  0.113969
GBM_grid_1_AutoML_1_20250321_14016_model_66              0.449968                0.491823  0.343517  0.118004
GBM_grid_1_AutoML_1_20250321_14016_model_166             0.449997                0.479858  0.342538  0.117333
GBM_grid_1_AutoML_1_20250321_14016_model_90              0.450274                0.494231  0.342837  0.117537
GBM_grid_1_AutoML_1_20250321_14016_model_9               0.450543                0.490097  0.339814  0.115474
GBM_grid_1_AutoML_1_20250321_14016_model_156             0.450552                0.493051  0.349713  0.122299
GBM_grid_1_AutoML_1_20250321_14016_model_95              0.450739                0.488073  0.34561   0.119446
GBM_grid_1_AutoML_1_20250321_14016_model_52              0.450848                0.488692  0.348153  0.121211
GBM_grid_1_AutoML_1_20250321_14016_model_169             0.450973                0.503894  0.349611  0.122228
GBM_grid_1_AutoML_1_20250321_14016_model_181             0.452015                0.486703  0.356328  0.126969
GBM_grid_1_AutoML_1_20250321_14016_model_155             0.452306                0.48447   0.351919  0.123847
GBM_grid_1_AutoML_1_20250321_14016_model_29              0.452637                0.516725  0.351918  0.123846
GBM_grid_1_AutoML_1_20250321_14016_model_5               0.454118                0.497274  0.346731  0.120222
GBM_grid_1_AutoML_1_20250321_14016_model_11              0.454565                0.491406  0.349599  0.122219
GBM_grid_1_AutoML_1_20250321_14016_model_26              0.455205                0.494378  0.349213  0.12195
GBM_grid_1_AutoML_1_20250321_14016_model_43              0.45601                 0.492247  0.35143   0.123503
GBM_grid_1_AutoML_1_20250321_14016_model_149             0.4563                  0.478844  0.347783  0.120953
GBM_grid_1_AutoML_1_20250321_14016_model_176             0.456731                0.488509  0.34872   0.121606
GBM_grid_1_AutoML_1_20250321_14016_model_78              0.457044                0.497259  0.350848  0.123094
GBM_grid_1_AutoML_1_20250321_14016_model_36              0.4573                  0.487673  0.34749   0.120749
GBM_grid_1_AutoML_1_20250321_14016_model_133             0.45765                 0.486387  0.354465  0.125645
GBM_grid_1_AutoML_1_20250321_14016_model_108             0.458279                0.488724  0.347533  0.120779
GBM_grid_1_AutoML_1_20250321_14016_model_117             0.458615                0.507862  0.353042  0.124639
GBM_5_AutoML_1_20250321_14016                            0.459431                0.469857  0.350133  0.122593
GBM_grid_1_AutoML_1_20250321_14016_model_97              0.45979                 0.489277  0.350558  0.122891
GBM_grid_1_AutoML_1_20250321_14016_model_175             0.460436                0.512538  0.357549  0.127842
GBM_grid_1_AutoML_1_20250321_14016_model_77              0.460493                0.498801  0.346006  0.11972
GBM_grid_1_AutoML_1_20250321_14016_model_129             0.460852                0.494878  0.352228  0.124064
GBM_grid_1_AutoML_1_20250321_14016_model_74              0.461587                0.495664  0.34823   0.121264
GBM_grid_1_AutoML_1_20250321_14016_model_130             0.462154                0.496055  0.352293  0.12411
GBM_grid_1_AutoML_1_20250321_14016_model_159             0.462206                0.49658   0.349928  0.12245
GBM_grid_1_AutoML_1_20250321_14016_model_154             0.46276                 0.489879  0.34899   0.121794
GBM_grid_1_AutoML_1_20250321_14016_model_110             0.463504                0.494823  0.350139  0.122597
GBM_grid_1_AutoML_1_20250321_14016_model_67              0.46378                 0.493088  0.344298  0.118541
GBM_grid_1_AutoML_1_20250321_14016_model_138             0.463939                0.492622  0.356112  0.126816
GBM_grid_1_AutoML_1_20250321_14016_model_112             0.464146                0.493062  0.343871  0.118248
GBM_grid_1_AutoML_1_20250321_14016_model_75              0.465044                0.487696  0.353217  0.124762
GBM_1_AutoML_1_20250321_14016                            0.465119                0.509305  0.353658  0.125074
GBM_grid_1_AutoML_1_20250321_14016_model_147             0.465375                0.493056  0.352311  0.124123
GBM_grid_1_AutoML_1_20250321_14016_model_106             0.465676                0.514788  0.358028  0.128184
GBM_grid_1_AutoML_1_20250321_14016_model_100             0.465719                0.495562  0.352473  0.124237
GBM_grid_1_AutoML_1_20250321_14016_model_179             0.46581                 0.490769  0.353409  0.124898
GBM_grid_1_AutoML_1_20250321_14016_model_173             0.466124                0.508974  0.35301   0.124616
GBM_grid_1_AutoML_1_20250321_14016_model_161             0.466644                0.495772  0.349048  0.121834
GBM_grid_1_AutoML_1_20250321_14016_model_85              0.466817                0.469445  0.338258  0.114419
GBM_grid_1_AutoML_1_20250321_14016_model_46              0.468076                0.478987  0.349066  0.121847
GBM_grid_1_AutoML_1_20250321_14016_model_23              0.468747                0.515511  0.357845  0.128053
GBM_grid_1_AutoML_1_20250321_14016_model_45              0.468985                0.501112  0.360533  0.129984
GBM_grid_1_AutoML_1_20250321_14016_model_144             0.470179                0.495225  0.35225   0.12408
GBM_grid_1_AutoML_1_20250321_14016_model_19              0.470549                0.48811   0.351339  0.123439
GBM_grid_1_AutoML_1_20250321_14016_model_25              0.470589                0.482235  0.354021  0.125331
GBM_grid_1_AutoML_1_20250321_14016_model_37              0.471074                0.493555  0.350757  0.123031
GBM_grid_1_AutoML_1_20250321_14016_model_171             0.47119                 0.494285  0.346679  0.120186
GBM_grid_1_AutoML_1_20250321_14016_model_99              0.472083                0.510958  0.35467   0.125791
GBM_grid_1_AutoML_1_20250321_14016_model_12              0.472435                0.489262  0.356152  0.126844
GBM_grid_1_AutoML_1_20250321_14016_model_139             0.473681                0.498091  0.353297  0.124819
GBM_grid_1_AutoML_1_20250321_14016_model_158             0.474381                0.476468  0.353435  0.124916
GBM_grid_1_AutoML_1_20250321_14016_model_22              0.474995                0.484399  0.349288  0.122002
GBM_grid_1_AutoML_1_20250321_14016_model_57              0.475995                0.498417  0.352766  0.124444
GBM_grid_1_AutoML_1_20250321_14016_model_3               0.476661                0.508444  0.360453  0.129926
GBM_grid_1_AutoML_1_20250321_14016_model_160             0.477138                0.521524  0.362156  0.131157
GBM_grid_1_AutoML_1_20250321_14016_model_127             0.477278                0.477337  0.353313  0.12483
GBM_grid_1_AutoML_1_20250321_14016_model_32              0.477469                0.489277  0.352887  0.124529
GBM_grid_1_AutoML_1_20250321_14016_model_73              0.478379                0.520684  0.362127  0.131136
GBM_grid_1_AutoML_1_20250321_14016_model_140             0.479212                0.477093  0.356779  0.127291
GBM_grid_1_AutoML_1_20250321_14016_model_116             0.479591                0.49973   0.359064  0.128927
GBM_grid_1_AutoML_1_20250321_14016_model_53              0.480569                0.476318  0.355438  0.126336
GBM_grid_1_AutoML_1_20250321_14016_model_81              0.480722                0.503952  0.364378  0.132771
GBM_grid_1_AutoML_1_20250321_14016_model_44              0.481331                0.491522  0.352185  0.124034
GBM_grid_1_AutoML_1_20250321_14016_model_111             0.481602                0.508023  0.352798  0.124466
GBM_grid_1_AutoML_1_20250321_14016_model_41              0.481916                0.474642  0.359866  0.129504
GBM_grid_1_AutoML_1_20250321_14016_model_27              0.482096                0.503308  0.352615  0.124337
GBM_grid_1_AutoML_1_20250321_14016_model_114             0.482115                0.485497  0.353386  0.124882
GBM_grid_1_AutoML_1_20250321_14016_model_120             0.482197                0.523762  0.360384  0.129877
GBM_grid_1_AutoML_1_20250321_14016_model_168             0.482302                0.470729  0.351833  0.123787
GBM_grid_1_AutoML_1_20250321_14016_model_119             0.482642                0.498016  0.352569  0.124305
GBM_grid_1_AutoML_1_20250321_14016_model_10              0.482982                0.486309  0.354984  0.126014
GBM_grid_1_AutoML_1_20250321_14016_model_20              0.482986                0.492031  0.353157  0.12472
GBM_grid_1_AutoML_1_20250321_14016_model_83              0.483815                0.515702  0.358799  0.128737
GBM_grid_1_AutoML_1_20250321_14016_model_96              0.483823                0.513189  0.363757  0.132319
GBM_grid_1_AutoML_1_20250321_14016_model_69              0.483975                0.505133  0.346282  0.119912
GBM_grid_1_AutoML_1_20250321_14016_model_87              0.484126                0.50417   0.364657  0.132975
GBM_grid_1_AutoML_1_20250321_14016_model_18              0.484413                0.499857  0.364641  0.132963
GBM_grid_1_AutoML_1_20250321_14016_model_150             0.484503                0.503601  0.368356  0.135686
GBM_grid_1_AutoML_1_20250321_14016_model_136             0.486034                0.517423  0.364425  0.132806
GBM_grid_1_AutoML_1_20250321_14016_model_16              0.487295                0.508317  0.365728  0.133757
GBM_grid_1_AutoML_1_20250321_14016_model_172             0.487724                0.497199  0.365754  0.133776
GBM_grid_1_AutoML_1_20250321_14016_model_33              0.48784                 0.516052  0.362417  0.131346
GBM_grid_1_AutoML_1_20250321_14016_model_178             0.488274                0.492368  0.368087  0.135488
GBM_grid_1_AutoML_1_20250321_14016_model_14              0.488806                0.51366   0.361953  0.13101
GBM_grid_1_AutoML_1_20250321_14016_model_55              0.489555                0.512198  0.363948  0.132458
GBM_grid_1_AutoML_1_20250321_14016_model_162             0.489875                0.506885  0.368989  0.136153
GBM_grid_1_AutoML_1_20250321_14016_model_104             0.490181                0.49193   0.353981  0.125302
GBM_grid_1_AutoML_1_20250321_14016_model_148             0.490434                0.486834  0.353966  0.125292
GBM_grid_1_AutoML_1_20250321_14016_model_38              0.491109                0.480253  0.351711  0.123701
GBM_grid_1_AutoML_1_20250321_14016_model_86              0.491389                0.510797  0.348002  0.121105
GBM_grid_1_AutoML_1_20250321_14016_model_145             0.494782                0.517672  0.366034  0.133981
GBM_grid_1_AutoML_1_20250321_14016_model_151             0.495837                0.512221  0.362657  0.13152
GBM_grid_1_AutoML_1_20250321_14016_model_125             0.497456                0.508478  0.366758  0.134511
DeepLearning_1_AutoML_1_20250321_14016                   0.501574                0.532114  0.356514  0.127102
GBM_grid_1_AutoML_1_20250321_14016_model_93              0.502647                0.490305  0.369317  0.136395
GBM_grid_1_AutoML_1_20250321_14016_model_65              0.504384                0.528349  0.373886  0.139791
GBM_grid_1_AutoML_1_20250321_14016_model_142             0.506818                0.513378  0.360819  0.130191
GBM_grid_1_AutoML_1_20250321_14016_model_113             0.508223                0.512932  0.374092  0.139945
GBM_grid_1_AutoML_1_20250321_14016_model_58              0.510632                0.515024  0.377342  0.142387
GBM_grid_1_AutoML_1_20250321_14016_model_40              0.510637                0.459361  0.353891  0.125239
GBM_grid_1_AutoML_1_20250321_14016_model_101             0.512101                0.500648  0.372304  0.13861
GBM_grid_1_AutoML_1_20250321_14016_model_59              0.513581                0.532882  0.375665  0.141124
GBM_grid_1_AutoML_1_20250321_14016_model_98              0.513872                0.512415  0.376674  0.141883
GBM_grid_1_AutoML_1_20250321_14016_model_51              0.515054                0.51551   0.376584  0.141816
GBM_grid_1_AutoML_1_20250321_14016_model_137             0.516753                0.497923  0.378055  0.142925
GBM_grid_1_AutoML_1_20250321_14016_model_128             0.518865                0.538003  0.374553  0.14029
GBM_grid_1_AutoML_1_20250321_14016_model_174             0.519442                0.50609   0.381243  0.145346
GBM_grid_1_AutoML_1_20250321_14016_model_102             0.5198                  0.521782  0.365391  0.13351
GBM_grid_1_AutoML_1_20250321_14016_model_49              0.520827                0.531597  0.38264   0.146413
GBM_grid_1_AutoML_1_20250321_14016_model_62              0.523013                0.470741  0.359647  0.129346
StackedEnsemble_AllModels_4_AutoML_1_20250321_14016      0.524497                0.575804  0.392823  0.15431
StackedEnsemble_AllModels_3_AutoML_1_20250321_14016      0.526205                0.575937  0.393516  0.154855
GBM_grid_1_AutoML_1_20250321_14016_model_152             0.527219                0.470177  0.356577  0.127147
GBM_grid_1_AutoML_1_20250321_14016_model_170             0.531515                0.525758  0.386198  0.149149
GBM_grid_1_AutoML_1_20250321_14016_model_115             0.533302                0.478537  0.362502  0.131408
GBM_grid_1_AutoML_1_20250321_14016_model_72              0.565691                0.518739  0.399641  0.159713
GBM_grid_1_AutoML_1_20250321_14016_model_164             0.567369                0.530917  0.401948  0.161562
GBM_grid_1_AutoML_1_20250321_14016_model_24              0.570966                0.52879   0.404409  0.163547
GBM_grid_1_AutoML_1_20250321_14016_model_107             0.576552                0.515282  0.40498   0.164009
XRT_1_AutoML_1_20250321_14016                            0.623414                0.482088  0.405738  0.164623
DeepLearning_grid_1_AutoML_1_20250321_14016_model_1      0.635095                0.554879  0.389415  0.151644
DeepLearning_grid_1_AutoML_1_20250321_14016_model_4      0.653635                0.560202  0.372681  0.138891
DeepLearning_grid_3_AutoML_1_20250321_14016_model_2      0.670773                0.544494  0.384583  0.147904
DeepLearning_grid_2_AutoML_1_20250321_14016_model_2      0.683951                0.566318  0.385527  0.148631
DeepLearning_grid_1_AutoML_1_20250321_14016_model_2      0.695366                0.571634  0.398314  0.158654
DeepLearning_grid_1_AutoML_1_20250321_14016_model_3      0.714658                0.606011  0.453829  0.205961
DRF_1_AutoML_1_20250321_14016                            0.732202                0.48763   0.395058  0.156071
DeepLearning_grid_1_AutoML_1_20250321_14016_model_7      0.752748                0.610142  0.445747  0.19869
DeepLearning_grid_1_AutoML_1_20250321_14016_model_5      0.755134                0.597291  0.443087  0.196326
DeepLearning_grid_2_AutoML_1_20250321_14016_model_1      0.779191                0.575523  0.435591  0.18974
DeepLearning_grid_1_AutoML_1_20250321_14016_model_6      0.85247                 0.542107  0.398618  0.158896
DeepLearning_grid_3_AutoML_1_20250321_14016_model_1      1.09997                 0.577232  0.536048  0.287348
GLM_1_AutoML_1_20250321_14016                            1.18779                 0.857179  0.636926  0.405675
[209 rows x 5 columns]

ModelMetricsMultinomialGLM: stackedensemble
** Reported on test data. **

MSE: 0.11174311883031816
RMSE: 0.33428000064364927
LogLoss: 0.3784270470272288
Null degrees of freedom: 3117
Residual degrees of freedom: 3017
Null deviance: 7818.674320837939
Residual deviance: 2359.8710652617992
AUC table was not computed: it is either disabled (model parameter 'auc_type' was set to AUTO or NONE) or the domain size exceeds the limit (maximum is 50 domains).
AUCPR table was not computed: it is either disabled (model parameter 'auc_type' was set to AUTO or NONE) or the domain size exceeds the limit (maximum is 50 domains).

Confusion Matrix: Row labels: Actual class; Column labels: Predicted class
EMPTY    I_FORM    JUMBO    PISTOL    SHOTGUN    SINGLEBACK    WILDCAT    Error      Rate
-------  --------  -------  --------  ---------  ------------  ---------  ---------  -----------
237      0         0        2         32         2             0          0.131868   36 / 273
0        96        2        10        2          93            0          0.527094   107 / 203
0        2         4        1         1          18            0          0.846154   22 / 26
1        19        1        23        6          75            0          0.816      102 / 125
21       0         0        8         1680       13            0          0.0243902  42 / 1,722
4        33        0        15        14         691           0          0.0871863  66 / 757
2        0         0        0         8          2             0          1          12 / 12
265      150       7        59        1743       894           0          0.124118   387 / 3,118

Top-7 Hit Ratios:
k    hit_ratio
---  -----------
1    0.875882
2    0.949006
3    0.983002
4    0.991982
5    0.997434
6    0.999679
7    1
Model saved to:  C:\Users\quort\OneDrive\Desktop\coding\model\StackedEnsemble_BestOfFamily_4_AutoML_1_20250321_14016
H2O session _sid_a895 closed.
'''