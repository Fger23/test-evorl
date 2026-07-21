from lerobot.datasets.aggregate import aggregate_datasets


inputs_dir = [
    "/nfs/lerobot/s101/datasets/evo_0714_insert_card_1/",
    "/nfs/lerobot/s101/datasets/evo_0714_insert_card_2/",
    "/nfs/lerobot/s101/datasets/evo_0714_insert_card_3/",
    "/nfs/lerobot/s101/datasets/evo_0715_insert_card_1/",

    ]
aggr_root = "/nfs/lerobot/s101/datasets/aggregation/evo-rl/insert_card_0714_15/"

# inputs_dir = [
#     "/nfs/lerobot/s101/datasets/0703_banknote_binding/",
#     "/nfs/lerobot/s101/datasets/0703_banknote_binding_1/",
#     "/nfs/lerobot/s101/datasets/0703_banknote_binding_2/",
#     "/nfs/lerobot/s101/datasets/0707_banknote_binding_1/",
#     "/nfs/lerobot/s101/datasets/0707_banknote_binding_2/",
#     "/nfs/lerobot/s101/datasets/0707_banknote_binding_3/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_1/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_2/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_3/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_4/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_5/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_6/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_7/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_8/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_9/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_10/",
#     "/nfs/lerobot/s101/datasets/0713_banknote_binding_11/",

#     ]
# # aggr_root = "/nfs/lerobot/s101/datasets/aggregation/evo-rl/banknote_binding_iteration_1/"
# aggr_root = "/nfs/lerobot/s101/datasets/aggregation/banknote_binding_iteration_0703_13/"

repo_ids = [""]*len(inputs_dir)
aggregate_datasets(
    # repo_ids=["classify_cylinder_and_cube_160", "classify_cylinder_and_cube_100_0303"],
    
    # aggr_repo_id="classify_cylinder_and_cube_260_0303_2",
    
    # roots = ["/data/s101/datasets/", "/data/s101/datasets/"],
    # aggr_root = "/data/s101/datasets/",
    # repo_ids = ["", ""],
    # aggr_repo_id = "",
    # roots = ["/data/s101/datasets/classify_cylinder_and_cube_160", "/data/s101/datasets/classify_cylinder_and_cube_100_0303"],
    # aggr_root = "/data/s101/datasets/classify_cylinder_and_cube_260_0303_2",
    # roots = ["/nfs/vcheck/tg/insert_card/", "/nfs/vcheck/tg/right_card_only_merged_0306/"],
    # aggr_root = "/nfs/vcheck/tg/aggregated/retrieve_and_insert_card_0314/",
    # repo_ids = ["", "", "", "", ""],
    # aggr_repo_id = "",
    # roots = ["/data/s101/datasets/0410_bimanual_banknotes_1_new/", "/data/s101/datasets/0410_bimanual_banknotes_2/", "/data/s101/datasets/0410_bimanual_banknotes_3/", 
    #          "/data/s101/datasets/0410_bimanual_banknotes_4/", "/data/s101/datasets/0410_bimanual_banknotes_5/"],
    # aggr_root = "/data/s101/datasets/aggregated/0410_bimanual_banknotes_1_5/",
    repo_ids = repo_ids,
    aggr_repo_id = "",
    roots = inputs_dir,
    aggr_root = aggr_root,
    data_files_size_in_mb=200,  # 每个数据文件最大200MB
    video_files_size_in_mb=500   # 每个视频文件最大500MB
)