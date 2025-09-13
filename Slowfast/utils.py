import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import numpy as np
from torchvision.transforms import v2
import shutil
class MaximalOptimizedVideoDatasetFixed(Dataset):
    """
    Dataset con máxima optimización - CORREGIDO y sin bugs
    """
    def __init__(self, root_dir, num_frames=32, img_size=224, transform=None, 
                 mode='train', overlap_ratio=0.3):
        
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.overlap_ratio = overlap_ratio
        
        # ✅ ELIMINADO: fps_standardization (causa problemas)
        # ✅ ELIMINADO: target_fps (no necesario)
        
        # Configuración conservadora pero efectiva
        if mode == 'train':
            self.max_clips_per_video = 4  # Reducido de 6 a 4
        elif mode == 'val':
            self.max_clips_per_video = 2  # Reducido de 4 a 2  
        else:  # test
            self.max_clips_per_video = 1  # Más conservador para test
        
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print(f"🔧 Inicializando {mode} dataset (CORREGIDO)...")
        self._build_robust_samples()
        print(f"✅ {mode} dataset: {len(self.samples)} samples")

    def _build_robust_samples(self):
        """Construye samples de forma robusta sin bugs"""
        
        total_videos = 0
        total_clips = 0
        valid_videos = 0
        
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            video_files = glob.glob(os.path.join(cls_dir, "*.mp4"))
            total_videos += len(video_files)
            
            for vf in video_files:
                label = self.class_to_idx[cls]
                
                # ✅ CORRECCIÓN: Verificar video antes de procesarlo
                if self._is_valid_video(vf):
                    valid_videos += 1
                    clips_for_video = self._calculate_safe_clips(vf)
                    total_clips += clips_for_video
                    
                    for clip_idx in range(clips_for_video):
                        self.samples.append((vf, label, clip_idx))
                else:
                    print(f"⚠️ Skipping invalid video: {os.path.basename(vf)}")
        
        print(f"   📹 Videos totales: {total_videos}")
        print(f"   ✅ Videos válidos: {valid_videos}")
        print(f"   🎬 Clips generados: {total_clips}")
        print(f"   📊 Clips/video: {total_clips/valid_videos:.2f}")

    def _is_valid_video(self, video_path):
        """Verifica que el video sea válido"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Test de lectura de primer frame
            ret, frame = cap.read()
            cap.release()
            
            # Verificaciones básicas
            return (frame_count > 0 and fps > 0 and ret and frame is not None)
            
        except Exception:
            return False

    def _calculate_safe_clips(self, video_path):
        """Calcula clips de forma segura"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if total_frames < self.num_frames:
                return 1  # Video corto: 1 clip
            else:
                # Video largo: calcular clips con overlap
                stride = max(1, int(self.num_frames * (1 - self.overlap_ratio)))
                possible_clips = (total_frames - self.num_frames) // stride + 1
                return min(possible_clips, self.max_clips_per_video)
                
        except Exception:
            return 1  # Fallback seguro

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, clip_idx = self.samples[idx]
        
        # ✅ CORRECCIÓN: Error handling sin recursión infinita
        max_retries = 3
        for attempt in range(max_retries):
            try:
                frames = self._load_clip_safe(video_path, clip_idx)
                
                if self.transform:
                    # ✅ CORRECCIÓN: Transform seguro frame por frame
                    transformed_frames = []
                    for frame in frames:
                        try:
                            transformed_frame = self.transform(frame)
                            transformed_frames.append(transformed_frame)
                        except Exception as e:
                            print(f"⚠️ Transform error: {e}")
                            # Usar frame normalizado como fallback
                            frame_norm = frame.float() / 255.0 if frame.max() > 1 else frame.float()
                            transformed_frames.append(frame_norm)
                    
                    frames = torch.stack(transformed_frames)
                else:
                    # Sin transform: normalizar a [0,1]
                    frames = torch.stack([f.float() / 255.0 if f.max() > 1 else f.float() for f in frames])
                
                frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
                return frames, label
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed for {video_path}: {e}")
                if attempt == max_retries - 1:
                    # Último intento: retornar datos dummy válidos
                    print(f"❌ Creating dummy data for {video_path}")
                    dummy_frames = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
                    return dummy_frames, label

    def _load_clip_safe(self, video_path, clip_idx):
        """Carga clip de forma completamente segura"""
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            # Video corto: cargar todo y hacer loop
            frames = self._load_all_frames_safe(cap)
            frames = self._safe_loop_pad(frames)
        else:
            # Video largo: extraer segmento específico
            frames = self._load_segment_safe(cap, total_frames, clip_idx)
        
        cap.release()
        return frames

    def _load_all_frames_safe(self, cap):
        """Carga todos los frames de forma segura"""
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                tensor = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
                frames.append(tensor)
                
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                continue
        
        return frames

    def _load_segment_safe(self, cap, total_frames, clip_idx):
        """Carga segmento específico de forma segura"""
        
        # Calcular inicio del segmento de forma conservadora
        if self.max_clips_per_video > 1:
            # Distribuir clips uniformemente sin overlap excesivo
            segment_length = total_frames // self.max_clips_per_video
            start_frame = min(clip_idx * segment_length, total_frames - self.num_frames)
        else:
            # Un solo clip: del centro
            start_frame = max(0, (total_frames - self.num_frames) // 2)
        
        # Saltar al frame inicial
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frames_needed = self.num_frames
        
        while len(frames) < frames_needed:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                tensor = torch.from_numpy(frame).permute(2, 0, 1)
                frames.append(tensor)
                
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                continue
        
        # Completar frames si es necesario
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                # Crear frame negro como último recurso
                black_frame = torch.zeros(3, self.img_size, self.img_size)
                frames.append(black_frame)
        
        return frames[:self.num_frames]

    def _safe_loop_pad(self, frames):
        """Loop seguro sin errores"""
        if len(frames) == 0:
            # Crear frames negros
            return [torch.zeros(3, self.img_size, self.img_size) for _ in range(self.num_frames)]
        
        target_len = self.num_frames
        out = []
        
        # Repetir frames de forma simple y segura
        while len(out) < target_len:
            for frame in frames:
                if len(out) >= target_len:
                    break
                out.append(frame.clone())
        
        return out[:target_len]


def get_corrected_optimized_loaders(dataset_dir, batch_size=4, num_frames=32, img_size=224,
                                   overlap_ratio=0.3):
    """
    Crea DataLoaders con dataset corregido
    """
    
    print("🛠️ Creando DataLoaders con dataset CORREGIDO...")
    
    # Transforms simplificados y robustos
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Crear datasets corregidos
    train_ds = MaximalOptimizedVideoDatasetFixed(
        os.path.join(dataset_dir, "train"),
        num_frames=num_frames,
        img_size=img_size,
        transform=train_transform,
        mode='train',
        overlap_ratio=overlap_ratio
    )
    
    val_ds = MaximalOptimizedVideoDatasetFixed(
        os.path.join(dataset_dir, "val"),
        num_frames=num_frames,
        img_size=img_size,
        transform=eval_transform,
        mode='val',
        overlap_ratio=overlap_ratio
    )
    
    test_ds = MaximalOptimizedVideoDatasetFixed(
        os.path.join(dataset_dir, "test"),
        num_frames=num_frames,
        img_size=img_size,
        transform=eval_transform,
        mode='test',
        overlap_ratio=overlap_ratio
    )

    # Crear DataLoaders con configuración estable
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)  # Menos workers para estabilidad
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)

    print(f"\n🛠️ DATASET CORREGIDO:")
    print(f"├── Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"├── Val: {len(val_ds)} samples ({len(val_loader)} batches)")
    print(f"└── Test: {len(test_ds)} samples ({len(test_loader)} batches)")
    
    total_samples = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"\n🎯 Total samples: {total_samples}")
    
    return train_loader, val_loader, test_loader

class IntelligentVideoSplit:
    """
    Split inteligente que maximiza la utilización de datos
    """
    def __init__(self, src_dir, dest_dir, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, 
                 stratify_by_duration=True, ensure_class_balance=True, seed=42):
        
        
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify_by_duration = stratify_by_duration
        self.ensure_class_balance = ensure_class_balance
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)

    def create_intelligent_split(self):
        """Crea split inteligente maximizando datos"""
        
        print("🧠 Creando split inteligente...")
        
        # Analizar dataset completo
        dataset_stats = self._analyze_full_dataset()
        
        # Optimizar ratios basado en estadísticas
        optimized_ratios = self._optimize_split_ratios(dataset_stats)
        
        # Crear splits estratificados
        self._create_stratified_splits(dataset_stats, optimized_ratios)
        
        # Generar reporte
        self._generate_split_report(dataset_stats, optimized_ratios)

    def _analyze_full_dataset(self):
        """Análisis completo del dataset"""
        
        stats = {}
        
        for class_name in os.listdir(self.src_dir):
            class_path = os.path.join(self.src_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            videos = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mkv'))]
            video_info = []
            
            for video in videos:
                video_path = os.path.join(class_path, video)
                info = self._get_video_info(video_path)
                info['filename'] = video
                video_info.append(info)
            
            stats[class_name] = {
                'videos': video_info,
                'count': len(videos),
                'durations': [v['duration'] for v in video_info],
                'frame_counts': [v['frames'] for v in video_info],
                'fps_values': [v['fps'] for v in video_info]
            }
        
        return stats

    def _get_video_info(self, video_path):
        """Obtiene información detallada del video"""
        
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Categorizar video
        if duration < 3:
            category = 'short'
        elif duration <= 8:
            category = 'medium'
        else:
            category = 'long'
        
        return {
            'frames': frame_count,
            'fps': fps,
            'duration': duration,
            'category': category,
            'potential_clips': max(1, frame_count // 32) if frame_count >= 32 else 1
        }

    def _optimize_split_ratios(self, dataset_stats):
        """Optimiza ratios de split basado en características del dataset"""
        
        # Analizar distribución de clips potenciales
        total_potential_clips = 0
        class_clip_counts = {}
        
        for class_name, stats in dataset_stats.items():
            class_clips = sum(v['potential_clips'] for v in stats['videos'])
            class_clip_counts[class_name] = class_clips
            total_potential_clips += class_clips
        
        print(f"📊 Clips potenciales totales: {total_potential_clips}")
        for class_name, count in class_clip_counts.items():
            print(f"  {class_name}: {count} clips")
        
        # Ajustar ratios para maximizar training data
        if total_potential_clips > 10000:
            # Dataset grande: más datos para validation/test
            return {'train': 0.70, 'val': 0.20, 'test': 0.10}
        elif total_potential_clips > 5000:
            # Dataset medio: balance estándar
            return {'train': 0.75, 'val': 0.15, 'test': 0.10}
        else:
            # Dataset pequeño: maximizar training
            return {'train': 0.80, 'val': 0.15, 'test': 0.05}

    def _create_stratified_splits(self, dataset_stats, ratios):
        """Crea splits estratificados"""
        
        for class_name, stats in dataset_stats.items():
            videos = stats['videos']
            
            # Estratificar por duración si está habilitado
            if self.stratify_by_duration:
                videos = self._stratify_by_duration(videos)
            else:
                random.shuffle(videos)
            
            # Calcular tamaños de splits
            n_total = len(videos)
            n_train = int(n_total * ratios['train'])
            n_val = int(n_total * ratios['val'])
            
            # Crear splits
            train_videos = videos[:n_train]
            val_videos = videos[n_train:n_train + n_val]
            test_videos = videos[n_train + n_val:]
            
            # Copiar archivos
            self._copy_split_files(class_name, {
                'train': train_videos,
                'val': val_videos,
                'test': test_videos
            })

    def _stratify_by_duration(self, videos):
        """Estratifica videos por duración para distribución balanceada"""
        
        # Separar por categorías
        short_videos = [v for v in videos if v['category'] == 'short']
        medium_videos = [v for v in videos if v['category'] == 'medium']
        long_videos = [v for v in videos if v['category'] == 'long']
        
        # Shuffle cada categoría
        random.shuffle(short_videos)
        random.shuffle(medium_videos)
        random.shuffle(long_videos)
        
        # Intercalar para distribución uniforme
        stratified = []
        max_len = max(len(short_videos), len(medium_videos), len(long_videos))
        
        for i in range(max_len):
            if i < len(short_videos):
                stratified.append(short_videos[i])
            if i < len(medium_videos):
                stratified.append(medium_videos[i])
            if i < len(long_videos):
                stratified.append(long_videos[i])
        
        return stratified

    def _copy_split_files(self, class_name, splits):
        """Copia archivos a splits correspondientes"""
        
        for split_name, videos in splits.items():
            split_dir = os.path.join(self.dest_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for video_info in videos:
                src_file = os.path.join(self.src_dir, class_name, video_info['filename'])
                dst_file = os.path.join(split_dir, video_info['filename'])
                shutil.copy2(src_file, dst_file)

    def _generate_split_report(self, dataset_stats, ratios):
        """Genera reporte detallado del split"""
        
        print(f"\n📋 REPORTE DE SPLIT OPTIMIZADO:")
        print(f"{'='*50}")
        
        for split_name, ratio in ratios.items():
            print(f"\n{split_name.upper()} ({ratio*100:.0f}%):")
            
            for class_name, stats in dataset_stats.items():
                n_videos = int(len(stats['videos']) * ratio)
                potential_clips = sum(v['potential_clips'] for v in stats['videos'][:n_videos])
                
                print(f"  {class_name}:")
                print(f"    Videos: {n_videos}")
                print(f"    Clips potenciales: {potential_clips}")



