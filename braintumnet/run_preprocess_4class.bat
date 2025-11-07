@echo off
echo ============================================================
echo Preprocessing BraTS Data to 4-Class Format
echo ============================================================
echo.
echo This will create 4-class labels: 0=BG, 1=NCR, 2=ED, 3=ET
echo Output directory: data/processed_multiclass_4class
echo.
echo Press Ctrl+C to cancel, or
pause

python scripts\preprocess_nifti_to_multiclass.py ^
    --nifti_dir "E:\thong\code\brain_segmen\braintumnet\data\raw\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData" ^
    --out_dir "E:\thong\code\brain_segmen\braintumnet\data\processed_multiclass_4class" ^
    --img_size 256 ^
    --num_folds 5 ^
    --seed 42

echo.
echo ============================================================
echo Preprocessing completed!
echo ============================================================
echo.
echo Next steps:
echo 1. Check data/processed_multiclass_4class for output
echo 2. Verify class_mapping.json shows 4 classes
echo 3. Run training: python scripts/train.py --model unetr --fold 0
echo.
pause
