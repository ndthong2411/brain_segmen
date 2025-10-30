"""
Convert BraTS H5 format to DICOM format
=========================================

Input: H5 files with image (H, W, 4) and mask (H, W, 3)
Output: DICOM files organized by patient and modality

This script converts H5 files to DICOM format for:
1. Clinical integration and PACS compatibility
2. Standardized medical imaging format
3. Better interoperability with medical software

Output structure:
    dicom_output/
    ├── Patient001/
    │   ├── FLAIR/
    │   │   ├── IM0001.dcm
    │   │   ├── IM0002.dcm
    │   │   └── ...
    │   ├── T1/
    │   ├── T1CE/
    │   ├── T2/
    │   └── SEG/
    ├── Patient002/
    └── ...

Usage:
    # Convert all H5 files
    python scripts/convert_h5_to_dicom.py \
        --h5_dir "E:\data\brats2020\h5_files" \
        --out_dir "E:\data\brats2020_dicom" \
        --patient_prefix "BraTS20_"

    # Convert specific patient
    python scripts/convert_h5_to_dicom.py \
        --h5_dir "E:\data\brats2020\h5_files" \
        --out_dir "E:\data\brats2020_dicom" \
        --max_files 100

    # With custom metadata
    python scripts/convert_h5_to_dicom.py \
        --h5_dir "E:\data\brats2020\h5_files" \
        --out_dir "E:\data\brats2020_dicom" \
        --study_description "Brain Tumor MRI" \
        --institution "Medical Center"

Author: BrainTumNet H5-to-DICOM Extension
Date: 2025-10-29
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ImplicitVRLittleEndian
except ImportError:
    print("Error: pydicom not installed")
    print("Install with: pip install pydicom")
    sys.exit(1)


# ============================================================
# DICOM UID Generation
# ============================================================

def generate_study_uid():
    """Generate unique Study Instance UID."""
    return generate_uid()


def generate_series_uid():
    """Generate unique Series Instance UID."""
    return generate_uid()


def generate_instance_uid():
    """Generate unique SOP Instance UID."""
    return generate_uid()


# ============================================================
# DICOM Metadata Templates
# ============================================================

def create_dicom_template(pixel_array, patient_id, study_uid, series_uid,
                          instance_uid, modality_name, instance_number,
                          study_description="Brain Tumor MRI Study",
                          institution="BrainTumNet Research"):
    """Create DICOM dataset with standard metadata.

    Args:
        pixel_array: (H, W) numpy array with pixel data
        patient_id: Patient ID string
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        instance_uid: SOP Instance UID
        modality_name: 'FLAIR', 'T1', 'T1CE', 'T2', or 'SEG'
        instance_number: Slice number (1-based)
        study_description: Study description
        institution: Institution name

    Returns:
        FileDataset: DICOM dataset ready to save
    """
    # Current datetime
    dt = datetime.datetime.now()
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S')

    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = instance_uid
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # Create main dataset
    ds = FileDataset(
        filename="temp.dcm",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    # Patient information
    ds.PatientName = f"Patient_{patient_id}"
    ds.PatientID = patient_id
    ds.PatientBirthDate = ''  # Unknown
    ds.PatientSex = ''  # Unknown

    # Study information
    ds.StudyInstanceUID = study_uid
    ds.StudyID = patient_id
    ds.StudyDate = date_str
    ds.StudyTime = time_str
    ds.StudyDescription = study_description
    ds.AccessionNumber = ''

    # Series information
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = get_series_number(modality_name)
    ds.SeriesDescription = get_series_description(modality_name)
    ds.Modality = 'MR' if modality_name != 'SEG' else 'SEG'

    # Instance information
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = instance_uid
    ds.InstanceNumber = instance_number

    # Image information
    ds.ImageType = ['DERIVED', 'PRIMARY', 'AXIAL']
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned

    # Spatial information
    ds.SliceThickness = '1.0'
    ds.PixelSpacing = ['1.0', '1.0']
    ds.SliceLocation = str(instance_number * 1.0)
    ds.ImagePositionPatient = ['0', '0', str(instance_number * 1.0)]
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']

    # Acquisition information
    ds.AcquisitionDate = date_str
    ds.AcquisitionTime = time_str
    ds.ContentDate = date_str
    ds.ContentTime = time_str

    # Equipment information
    ds.Manufacturer = 'BrainTumNet'
    ds.InstitutionName = institution
    ds.ManufacturerModelName = 'Virtual MRI Scanner'
    ds.SoftwareVersions = '1.0'

    # Convert pixel array to uint16
    if pixel_array.dtype != np.uint16:
        # Scale to 0-65535 range
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_scaled = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 65535).astype(np.uint16)
        else:
            pixel_scaled = np.zeros_like(pixel_array, dtype=np.uint16)
    else:
        pixel_scaled = pixel_array

    # Set pixel data
    ds.PixelData = pixel_scaled.tobytes()

    # Rescale parameters (to convert back to original values)
    ds.RescaleIntercept = str(pixel_min)
    ds.RescaleSlope = str((pixel_max - pixel_min) / 65535.0 if pixel_max > pixel_min else 1.0)
    ds.RescaleType = 'US'  # Unspecified

    return ds


def get_series_number(modality_name):
    """Get series number based on modality."""
    series_map = {
        'FLAIR': 1,
        'T1': 2,
        'T1CE': 3,
        'T2': 4,
        'SEG': 5
    }
    return series_map.get(modality_name, 99)


def get_series_description(modality_name):
    """Get series description based on modality."""
    desc_map = {
        'FLAIR': 'FLAIR - Fluid Attenuated Inversion Recovery',
        'T1': 'T1 - Pre-Contrast',
        'T1CE': 'T1CE - Post-Contrast (Gd)',
        'T2': 'T2 - Weighted',
        'SEG': 'Segmentation Mask'
    }
    return desc_map.get(modality_name, 'Unknown Sequence')


# ============================================================
# H5 to DICOM Conversion
# ============================================================

def load_h5_data(h5_path):
    """Load H5 file and return image and mask.

    Args:
        h5_path: Path to H5 file

    Returns:
        image: (H, W, 4) numpy array - 4 modalities
        mask: (H, W, 3) numpy array - 3 binary channels
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]  # (H, W, 4)
            mask = f['mask'][:]    # (H, W, 3)
        return image, mask
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None, None


def convert_mask_to_single_channel(mask_3ch):
    """Convert 3-channel mask to single-channel for DICOM SEG.

    Args:
        mask_3ch: (H, W, 3) binary mask

    Returns:
        mask_single: (H, W) uint16 with encoded classes
            0 = Background
            1 = Tumor Core (from channel 1)
            2 = Edema (from channel 2)
    """
    H, W, C = mask_3ch.shape
    mask_single = np.zeros((H, W), dtype=np.uint16)

    # Encode: Channel 2 (Edema) → 2, Channel 1 (TC) → 1
    mask_single[mask_3ch[:, :, 2] > 0] = 2
    mask_single[mask_3ch[:, :, 1] > 0] = 1

    return mask_single


def convert_h5_to_dicom_patient(h5_path, out_dir, patient_id,
                                 study_description="Brain Tumor MRI Study",
                                 institution="BrainTumNet Research"):
    """Convert a single H5 file to DICOM series.

    Args:
        h5_path: Path to H5 file
        out_dir: Output directory
        patient_id: Patient ID
        study_description: Study description
        institution: Institution name

    Returns:
        success: True if successful, False otherwise
    """
    # Load H5 data
    image, mask = load_h5_data(h5_path)

    if image is None:
        return False

    # Extract slice info from filename
    # Example: volume_1_slice_50.h5 → vol1, slice 50
    fname = Path(h5_path).stem
    parts = fname.split('_')

    if len(parts) >= 4 and parts[0] == 'volume':
        volume_id = f"vol{parts[1]}"
        slice_idx = int(parts[3])
    else:
        # Fallback: use patient_id
        volume_id = patient_id
        slice_idx = 1

    # Create patient directory
    patient_dir = Path(out_dir) / f"{patient_id}_{volume_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Generate Study UID (same for all series of this patient)
    study_uid = generate_study_uid()

    # Modalities
    modalities = ['FLAIR', 'T1', 'T1CE', 'T2']

    # Convert each modality
    for mod_idx, mod_name in enumerate(modalities):
        # Create series directory
        series_dir = patient_dir / mod_name
        series_dir.mkdir(exist_ok=True)

        # Generate Series UID (unique for each modality)
        series_uid = generate_series_uid()

        # Extract modality image
        img_2d = image[:, :, mod_idx].astype(np.float32)

        # Generate Instance UID
        instance_uid = generate_instance_uid()

        # Create DICOM dataset
        ds = create_dicom_template(
            pixel_array=img_2d,
            patient_id=f"{patient_id}_{volume_id}",
            study_uid=study_uid,
            series_uid=series_uid,
            instance_uid=instance_uid,
            modality_name=mod_name,
            instance_number=slice_idx,
            study_description=study_description,
            institution=institution
        )

        # Save DICOM file
        dcm_filename = f"IM{slice_idx:04d}.dcm"
        dcm_path = series_dir / dcm_filename
        ds.save_as(str(dcm_path), write_like_original=False)

    # Convert segmentation mask
    seg_dir = patient_dir / "SEG"
    seg_dir.mkdir(exist_ok=True)

    series_uid = generate_series_uid()
    instance_uid = generate_instance_uid()

    # Convert mask to single channel
    mask_single = convert_mask_to_single_channel(mask)

    # Create DICOM segmentation
    ds = create_dicom_template(
        pixel_array=mask_single,
        patient_id=f"{patient_id}_{volume_id}",
        study_uid=study_uid,
        series_uid=series_uid,
        instance_uid=instance_uid,
        modality_name='SEG',
        instance_number=slice_idx,
        study_description=study_description,
        institution=institution
    )

    # Save segmentation DICOM
    dcm_filename = f"SEG{slice_idx:04d}.dcm"
    dcm_path = seg_dir / dcm_filename
    ds.save_as(str(dcm_path), write_like_original=False)

    return True


# ============================================================
# Batch Conversion
# ============================================================

def batch_convert_h5_to_dicom(h5_dir, out_dir, patient_prefix="Patient",
                               study_description="Brain Tumor MRI Study",
                               institution="BrainTumNet Research",
                               max_files=None):
    """Convert multiple H5 files to DICOM format.

    Args:
        h5_dir: Directory containing H5 files
        out_dir: Output directory for DICOM files
        patient_prefix: Prefix for patient IDs
        study_description: Study description
        institution: Institution name
        max_files: Maximum number of files to convert (None = all)

    Returns:
        stats: Dict with conversion statistics
    """
    h5_dir = Path(h5_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all H5 files
    h5_files = sorted(list(h5_dir.glob("*.h5")))

    if max_files:
        h5_files = h5_files[:max_files]

    print(f"Found {len(h5_files)} H5 files")

    # Group files by volume
    volume_groups = {}
    for h5_path in h5_files:
        fname = h5_path.stem
        parts = fname.split('_')

        if len(parts) >= 2 and parts[0] == 'volume':
            volume_id = f"vol{parts[1]}"
        else:
            volume_id = fname

        if volume_id not in volume_groups:
            volume_groups[volume_id] = []
        volume_groups[volume_id].append(h5_path)

    print(f"Grouped into {len(volume_groups)} patients/volumes")

    # Convert each volume
    success_count = 0
    fail_count = 0

    for vol_idx, (volume_id, h5_paths) in enumerate(tqdm(volume_groups.items(),
                                                           desc="Converting patients")):
        patient_id = f"{patient_prefix}{vol_idx+1:03d}"

        for h5_path in h5_paths:
            success = convert_h5_to_dicom_patient(
                h5_path=h5_path,
                out_dir=out_dir,
                patient_id=patient_id,
                study_description=study_description,
                institution=institution
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

    stats = {
        'total_h5_files': len(h5_files),
        'total_patients': len(volume_groups),
        'success': success_count,
        'failed': fail_count,
        'success_rate': success_count / len(h5_files) * 100 if len(h5_files) > 0 else 0
    }

    return stats


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Convert BraTS H5 to DICOM format")
    parser.add_argument("--h5_dir", type=str, required=True,
                       help="Directory containing H5 files")
    parser.add_argument("--out_dir", type=str, default="data/dicom_output",
                       help="Output directory for DICOM files")
    parser.add_argument("--patient_prefix", type=str, default="BraTS20_",
                       help="Prefix for patient IDs")
    parser.add_argument("--study_description", type=str,
                       default="Brain Tumor MRI Study",
                       help="Study description in DICOM")
    parser.add_argument("--institution", type=str,
                       default="BrainTumNet Research",
                       help="Institution name")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of H5 files to convert (for testing)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("H5 to DICOM Converter - BrainTumNet")
    print(f"{'='*60}\n")

    print(f"Input H5 directory: {args.h5_dir}")
    print(f"Output DICOM directory: {args.out_dir}")
    print(f"Patient prefix: {args.patient_prefix}")
    print(f"Study description: {args.study_description}")
    print(f"Institution: {args.institution}")

    if args.max_files:
        print(f"Max files to convert: {args.max_files}")

    print(f"\n{'='*60}")
    print("Starting conversion...")
    print(f"{'='*60}\n")

    # Convert
    stats = batch_convert_h5_to_dicom(
        h5_dir=args.h5_dir,
        out_dir=args.out_dir,
        patient_prefix=args.patient_prefix,
        study_description=args.study_description,
        institution=args.institution,
        max_files=args.max_files
    )

    # Print statistics
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}\n")

    print(f"Total H5 files: {stats['total_h5_files']}")
    print(f"Total patients: {stats['total_patients']}")
    print(f"Successfully converted: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")

    print(f"\nOutput directory: {args.out_dir}")
    print("\nDICOM structure:")
    print("  dicom_output/")
    print("  ├── Patient001_vol1/")
    print("  │   ├── FLAIR/")
    print("  │   ├── T1/")
    print("  │   ├── T1CE/")
    print("  │   ├── T2/")
    print("  │   └── SEG/")
    print("  ├── Patient002_vol2/")
    print("  └── ...")

    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("\n1. Verify DICOM files:")
    print("   python -c \"import pydicom; ds = pydicom.dcmread('dicom_output/Patient001_vol1/FLAIR/IM0001.dcm'); print(ds)\"")
    print("\n2. Convert DICOM back to PNG for training:")
    print("   python scripts/preprocess_dicom_to_multiclass.py \\")
    print("       --dicom_dir data/dicom_output \\")
    print("       --out_dir data/processed_multiclass_from_dicom")
    print("\n3. View DICOM in medical viewer:")
    print("   - Use MicroDicom, RadiAnt, or other DICOM viewers")
    print("   - Load from: dicom_output/Patient001_vol1/")


if __name__ == "__main__":
    main()
