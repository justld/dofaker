# python run_faceswapper.py --source docs/test/multi.png --dst_face_paths docs/test/dst1.png --src_face_paths docs/test/trump.jpg

# python run_faceswapper.py --source docs/test/multi.png --dst_face_paths docs/test/dst1.png docs/test/dst2.png --src_face_paths docs/test/trump.jpg docs/test/taitan.jpeg

# python run_faceswapper.py --source docs/test/multi.png --dst_face_paths docs/test/dst1.png docs/test/dst2.png --src_face_paths docs/test/trump.jpg docs/test/taitan.jpeg --use_enhancer

python run_faceswapper.py --source docs/test/multi.png --dst_face_paths docs/test/dst1.png docs/test/dst2.png --src_face_paths docs/test/trump.jpg docs/test/taitan.jpeg --use_enhancer --use_sr --sr_scale 2.0

# python run_posetransfer.py --source docs/test/condition.jpg --target docs/test/target_pose_reference.jpg

# python run_posetransfer.py --source docs/test/condition.jpg --target docs/test/target_pose_reference.jpg --use_enhancer

python run_posetransfer.py --source docs/test/condition.jpg --target docs/test/target_pose_reference.jpg --use_enhancer --use_sr --sr_scale 2.0
