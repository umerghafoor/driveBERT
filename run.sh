# python inference_action.py \
# --config configs/action/MB_train_NTU60_xsub.yaml \
# --checkpoint checkpoint/action/MB_train_NTU60_xsub/best_epoch.bin \
# --input samples/sample_skeleton.pkl


# Action Recognition Inference
# python infer_action.py --config configs/action/MB_train_NTU60_xsub.yaml --checkpoint checkpoint/action/MB_train_NTU60_xsub/best_epoch.bin --input result/X3D.npy


# Video Inference 2D Pose
# python infer_wild.py --vid_path dta/truck_driver.mp4 --json_path dta/alphapose-results.json --out_path "result"

# Video Inference 3D Pose
python infer_wild_mesh.py --vid_path dta/truck_driver.mp4 --json_path dta/alphapose-results.json --out_path result_e
