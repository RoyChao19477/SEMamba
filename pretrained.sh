CUDA_VISIBLE_DEVICES='0' python inference.py \
   --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   --output_folder results \
   --checkpoint_file ckpts/SEMamba_advanced.pth  \
   --config recipes/SEMamba_advanced/SEMamba_advanced.yaml \
   --post_processing_PCS False \
   
