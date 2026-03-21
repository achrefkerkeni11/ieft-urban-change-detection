import os, io
import numpy as np
import pyarrow as pa
from PIL import Image

def make_example(idx):
    # Create a random 384x384 RGB image (like the model expects)
    arr = (np.random.rand(384, 384, 3) * 255).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")

    # Convert image to bytes (JPEG) because BaseDataset expects bytes in "image" column
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    # Captions must be a LIST of strings (because BaseDataset expects list<string>)
    captions = [f"dummy caption {idx}", "an aerial photo of a city"]
    path = f"dummy_{idx}.jpg"
    return img_bytes, path, captions

def write_arrow(out_path, n):
    images, paths, captions = [], [], []
    for i in range(n):
        img_bytes, path, caps = make_example(i)
        images.append(img_bytes)
        paths.append(path)
        captions.append(caps)

    table = pa.table({
        "image": pa.array(images, type=pa.binary()),
        "path": pa.array(paths, type=pa.string()),
        "caption": pa.array(captions, type=pa.list_(pa.string())),
    })

    # Write as Arrow IPC file (RecordBatchFileReader reads this format)
    with pa.OSFile(out_path, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write(table)

if __name__ == "__main__":
    os.makedirs("dummy_data", exist_ok=True)

    # Names must match what RSICD dataset class expects
    write_arrow("dummy_data/rsicd_caption_karpathy_new_train.arrow", n=2)
    write_arrow("dummy_data/rsicd_caption_karpathy_new_val.arrow", n=1)
    write_arrow("dummy_data/rsicd_caption_karpathy_new_test.arrow", n=1)

    print("Dummy RSICD arrow files created in dummy_data/")