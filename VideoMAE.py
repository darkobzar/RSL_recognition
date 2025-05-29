from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms.functional as TF
from collections import defaultdict


def collate_fn(batch):
    clips, labels = zip(*batch)
    clips = torch.stack(clips)
    labels = torch.tensor(labels)
    return clips, labels

class MixedGestureDataset(Dataset):
    def __init__(self, video_frames_dir, video_ann_path,
                 image_dir, image_ann_path,
                 clip_len=8, transform=None, crop_bbox=True):

        self.clip_len = clip_len
        self.transform = transform
        self.crop_bbox = crop_bbox


        with open(video_ann_path) as f:
            self.video_coco = json.load(f)
        self.video_image_id_to_file = {img['id']: img['file_name'] for img in self.video_coco['images']}
        self.video_annotations = self.video_coco['annotations']
        self.video_label_map = {cat['id']: cat['name'] for cat in self.video_coco['categories']}


        with open(image_ann_path) as f:
            self.image_coco = json.load(f)
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.image_coco['images']}
        self.image_annotations = self.image_coco['annotations']
        self.image_label_map = {cat['id']: cat['name'] for cat in self.image_coco['categories']}


        all_labels = set(self.video_label_map.values()) | set(self.image_label_map.values())
        self.label2id = {label: i for i, label in enumerate(sorted(all_labels))}


        
        self.video_bboxes = defaultdict(list)
        for ann in self.video_annotations:
            fname = self.video_image_id_to_file[ann['image_id']]
            self.video_bboxes[fname].append({"category_id": ann['category_id'], "bbox": ann['bbox']})

        self.image_bboxes = defaultdict(list)
        for ann in self.image_annotations:
            fname = self.image_id_to_file[ann['image_id']]
            self.image_bboxes[fname].append({"category_id": ann['category_id'], "bbox": ann['bbox']})


        self.video_samples = self.build_video_clips(video_frames_dir)
        self.image_samples = self.build_image_clips(image_dir)

        self.samples = self.video_samples + self.image_samples

    def build_video_clips(self, frames_dir):
        frame_label_pairs = []
        for ann in self.video_annotations:
            img_id = ann['image_id']
            label = self.video_label_map[ann['category_id']]
            file_name = self.video_image_id_to_file[img_id]
            frame_label_pairs.append((file_name, label))

        frame_label_pairs.sort()
        clips = []
        for i in range(len(frame_label_pairs) - self.clip_len + 1):
            window = frame_label_pairs[i:i + self.clip_len]
            labels_in_window = [label for _, label in window]
            if len(set(labels_in_window)) == 1:
                paths = []
                for file_name, _ in window:
                    path = os.path.join(frames_dir, file_name)
                    if not os.path.exists(path):
                        break
                    paths.append(path)
                if len(paths) == self.clip_len:
                    clips.append((paths, self.label2id[labels_in_window[0]], labels_in_window[0]))
        return clips

    def build_image_clips(self, image_dir):
        clips = []
        for ann in self.image_annotations:
            file_name = self.image_id_to_file[ann['image_id']]
            label = self.image_label_map[ann['category_id']]
            path = os.path.join(image_dir, file_name)
            if os.path.exists(path):
                clips.append(([path] * self.clip_len, self.label2id[label], label))
        return clips

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label_id, label_str = self.samples[idx]
        clip = []

        for path in frame_paths:
            fname = os.path.basename(path)
            img = Image.open(path).convert("RGB")
            
            if self.crop_bbox:
                bboxes_dict = self.video_bboxes if fname in self.video_bboxes else self.image_bboxes
                if fname in bboxes_dict:
                    for ann in bboxes_dict[fname]:
                        cat_id = ann['category_id']
                        ann_label = self.video_label_map.get(cat_id) or self.image_label_map.get(cat_id)
                        if ann_label == label_str:
                            x, y, w, h = ann['bbox']
                            img = img.crop((int(x), int(y), int(x + w), int(y + h)))
                            break

            if self.transform:
                img = self.transform(img)
            clip.append(img)

        clip_tensor = torch.stack(clip) 
        return clip_tensor, label_id


class VideoGestureDataset(Dataset):
    def __init__(self, frames_dir, ann_path, clip_len=8, transform=None, label2id=None, crop_bbox=True):
        self.frames_dir = frames_dir
        self.clip_len = clip_len
        self.transform = transform
        self.crop_bbox = crop_bbox

        with open(ann_path, 'r') as f:
            coco = json.load(f)

        self.image_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
        self.label_map = {cat['id']: cat['name'] for cat in coco['categories']}
        self.annotations = coco['annotations']

        self.bboxes_by_image = defaultdict(list)
        label_to_files = defaultdict(list)
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_file:
                continue
            fname = self.image_id_to_file[img_id]
            label = self.label_map[ann['category_id']]
            label_to_files[label].append(fname)
            self.bboxes_by_image[fname].append({
                "category_id": ann['category_id'],
                "bbox": ann['bbox']
            }) 
        self.label2id = label2id
        self.samples = []
        for label, files in label_to_files.items():
            clips = self.build_clips(files, clip_len=self.clip_len)
            for clip_files in clips:
                clip_paths = [os.path.join(self.frames_dir, f) for f in clip_files]
                if all(os.path.exists(p) for p in clip_paths):
                    self.samples.append((clip_paths, self.label2id[label], label))

    def build_clips(self, files, clip_len):
        files = sorted(files)
        clips = []
        n = len(files)
        for i in range(n - clip_len + 1):
            clip_files = files[i:i + clip_len]
            clips.append(clip_files)
        return clips

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label_id, label_str = self.samples[idx]
        clip = []

        for path in frame_paths:
            fname = os.path.basename(path)
            img = Image.open(path).convert("RGB")

            if self.crop_bbox and fname in self.bboxes_by_image:
                for ann in self.bboxes_by_image[fname]:
                    ann_label = self.label_map.get(ann['category_id'])
                    if ann_label == label_str:
                        x, y, w, h = ann['bbox']
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        img = img.crop((x1, y1, x2, y2))
                        break 
            if self.transform:
                img = self.transform(img)
            clip.append(img)

        clip_tensor = torch.stack(clip) 
        return clip_tensor, label_id


def train(model, train_loader, val_loader, device, epochs=10, lr=1e-5, log_dir="runs/experiment1"):

    writer = SummaryWriter(log_dir=log_dir)
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay = 0.01)
    criterion = CrossEntropyLoss()

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")

        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            global_step += 1

            if global_step % 10 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/Accuracy", correct / total, global_step)

            pbar.set_postfix(loss=loss.item(), accuracy=correct / total)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss}, Accuracy: {epoch_acc:.4f}")
        torch.save(model.state_dict(), f'{epoch+1}.pth')
        writer.add_scalar("Epoch/Train_Loss", epoch_loss, epoch)
        writer.add_scalar("Epoch/Train_Accuracy", epoch_acc, epoch)

        val_loss, val_acc = evaluate_test(model, val_loader, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss}, Accuracy: {val_acc:.4f}")
        writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
        writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)

    writer.close()


def evaluate_test(model, data_loader, device):
    
    model.eval()
    correct = 0
    total = 0

    preds = []
    labels = []

    with torch.no_grad():
        for inputs, lbl in tqdm(data_loader):
            inputs = inputs.to(device)
            lbl = lbl.to(device)

            outputs = model(inputs).logits

            pr = torch.argmax(outputs, dim=1)
            correct += (pr == lbl).sum().item()
            total += lbl.size(0)

            preds.extend(pr.cpu().numpy())
            labels.extend(lbl.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    print(accuracy)
    print(precision)
    print(recall)
    print(f1)

    return accuracy, precision, recall, f1



def plot_class_distribution(dataset, id2label, split=''):

    labels = [label for _,_, label in dataset.samples]
    label_counts = Counter(labels)

    sorted_ids = sorted(label_counts.keys())
    class_names = [id2label[i] for i in sorted_ids]
    counts = [label_counts[i] for i in sorted_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Класс")
    plt.ylabel("Количество клипов")
    plt.title(f"Распределение классов в {split} датасете")
    plt.tight_layout()
    plt.show()


def show_clip_from_dataset(dataset, idx):
    clip_tensor, label = dataset[idx]

    id2label = {v: k for k, v in dataset.label2id.items()}
    print(f"Label name: {id2label[label]}")

    clip = clip_tensor.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]

    fig, axs = plt.subplots(1, clip.shape[0], figsize=(15, 3))
    for i in range(clip.shape[0]):
        axs[i].imshow(clip[i])
        axs[i].axis('off')
        axs[i].set_title(f"Frame {i}")
    plt.tight_layout()
    plt.show()