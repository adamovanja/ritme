# import os

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.metrics import root_mean_squared_error

# plt.rcParams.update({"font.family": "DejaVu Sans"})
# plt.style.use("seaborn-v0_8-pastel")

# # custom color map
# color_map = {
#     "train": "lightskyblue",
#     "test": "peachpuff",
#     "rmse_train": "lightskyblue",
#     "rmse_val": "plum",
# }


# def calculate_rmse(pred_df):
#     rmse_scores = {}
#     for split in pred_df["split"].unique():
#         pred_split = pred_df[pred_df["split"] == split].copy()
#         rmse = root_mean_squared_error(
#             pred_split["true"].values, pred_split["pred"].values
#         )
#         rmse_scores[split] = rmse
#     return rmse_scores


# def plot_rmse_over_experiments(preds_dic, save_loc, dpi=400):
#     if not os.path.exists(save_loc):
#         os.makedirs(save_loc)

#     rmse_dic = {}
#     for model_type, pred_df in preds_dic.items():
#         rmse_dic[model_type] = calculate_rmse(pred_df)

#     plt.figure(dpi=dpi)  # Increase the resolution by setting a higher dpi
#     rmse_df = pd.DataFrame(rmse_dic).T
#     rmse_df = rmse_df[
#         sorted(rmse_df.columns, key=lambda x: 0 if "train" in x else 1)
#     ]  # Enforce column order
#     rmse_df.plot(
#         kind="bar",
#         title="Overall",
#         ylabel="RMSE",
#         color=[color_map.get(col, "gray") for col in rmse_df.columns],
#     )
#     path_to_save = os.path.join(save_loc, "rmse_over_experiments_train_test.png")
#     plt.tight_layout()
#     plt.savefig(path_to_save, dpi=dpi)


# def plot_rmse_over_target_bins(preds_dic, ls_model_types, save_loc, dpi=300):
#     """
#     Plot RMSE over true target bins for each model type in ls_model_types.
#     """
#     for model_type in ls_model_types:
#         pred_df = preds_dic[model_type]
#         split = None

#         # Bin true columns
#         pred_df["group"] = np.round(pred_df["true"], 0).astype(int)

#         # Calculate RMSE for each group
#         grouped_ser = pred_df.groupby(["group"]).apply(calculate_rmse)
#         grouped_df = grouped_ser.apply(pd.Series)
#         if split is not None:
#             grouped_df = grouped_df[[split]].copy()

#         # Enforce column order
#         grouped_df = grouped_df[
#             sorted(grouped_df.columns, key=lambda x: 0 if "train" in x else 1)
#         ]

#         # Plot
#         plt.figure(dpi=dpi)
#         grouped_df.plot(
#             kind="bar",
#             title=f"Model: {model_type}",
#             ylabel="RMSE",
#             figsize=(10, 5),
#             color=[color_map.get(col, "gray") for col in grouped_df.columns],
#         )
#         path_to_save = os.path.join(
#             save_loc, f"rmse_over_time_train_test_{model_type}.png"
#         )
#         plt.savefig(path_to_save, dpi=dpi)
