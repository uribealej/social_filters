import re
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from pathlib import Path

def extract_aux_trigger_frames(tiff_path: Path, n_channels: int = 2):
    """
    Extract frame numbers and aux trigger values from a volumetric ScanImage TIFF stack.

    Parameters:
        tiff_path (Path): Full path to the TIFF stack.
        n_channels (int): Number of auxTrigger channels to extract.

    Returns:
        dict: Dictionary with trigger names as keys and lists of (frame, value) for nonzero triggers.
    """
    trigger_frames = {f'auxTrigger{ch}': [] for ch in range(n_channels)}

    with ScanImageTiffReader(str(tiff_path)) as tif:
        for frame_idx in range(len(tif)):
            metadata = tif.description(frame_idx)

            for ch in range(n_channels):
                key = f'auxTrigger{ch}'
                match = re.search(rf"{key} = \[(.*?)\]", metadata, re.DOTALL)
                if match:
                    values_str = match.group(1)
                    values = np.fromstring(values_str, sep=',')
                    # Store only if nonzero (trigger event occurred)
                    if np.any(values > 0):
                        trigger_frames[key].append((frame_idx, values))
                else:
                    print(f"{key} not found in frame {frame_idx}")

    return trigger_frames


# --- Example Usage ---
tiff_file = Path(r'E:/Matilde/2p_data/f30/00_raw/f30_00002.tif')
trigger_data = extract_aux_trigger_frames(tiff_file)

# Print results
for key, events in trigger_data.items():
    print(f"\n{key} triggered in {len(events)} frames:")
    for frame_idx, values in events:
        print(f"  Frame {frame_idx}: {values}")