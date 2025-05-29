import cv2
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from collections import Counter
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=1):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()


def get_bbox(input_folder, model, conf_threshold=0.5):

    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png'))])
    
    bbox = {}
    
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        
        frame = cv2.imread(frame_path)
        results = model(frame, conf=conf_threshold)
        
        for result in results:
            x1, y1, x2, y2 = map(int, result.boxes[0].xyxy[0].tolist())
            bbox[frame_file] = [x1, y1, x2, y2] 
    
    return bbox
    

class VideoDataset(Dataset):
    def __init__(self, frames_dir, anno, clip_len=8, transform=None, crop_bbox=True):

        self.frames_dir = frames_dir
        self.bbox_annotations = anno
        self.clip_len = clip_len
        self.transform = transform
        self.crop_bbox = crop_bbox
        
        self.frame_files = sorted([f for f in anno.keys() 
                                 if os.path.exists(os.path.join(frames_dir, f))])
        
        self.clips = self.build_clips(self.frame_files, self.clip_len)

    def build_clips(self, files, clip_len):

        clips = []
        n = len(files)
        for i in range(n - clip_len + 1):
            clip_files = files[i:i + clip_len]
            clips.append(clip_files)
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_files = self.clips[idx]
        clip = []

        for frame_file in clip_files:
            frame_path = os.path.join(self.frames_dir, frame_file)
            img = Image.open(frame_path).convert("RGB")

            if self.crop_bbox and frame_file in self.bbox_annotations:
                x1, y1, x2, y2 = self.bbox_annotations[frame_file]
                img = img.crop((x1, y1, x2, y2))
            
            if self.transform:
                img = self.transform(img)
            
            clip.append(img)

        clip_tensor = torch.stack(clip)

        return clip_tensor

def evaluate(model, data_loader, device):
    
    model.eval()

    preds = []

    with torch.no_grad():
        for inputs in tqdm(data_loader):
            inputs = inputs.to(device)

            outputs = model(inputs).logits

            pr = torch.argmax(outputs, dim=1)
            preds.extend(pr.cpu().numpy())

    return Counter(preds).most_common(1)[0][0]