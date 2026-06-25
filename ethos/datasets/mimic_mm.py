import torch as th
from .base import InferenceDataset

class MMPFSDataset(InferenceDataset):
    def __init__(self, data, encode, block_size):
        super().__init__(data, encode, block_size)
        mm_id = 2034  # 你搜到的多发性骨髓瘤 Token ID
        
        # 找到所有 MM Token 的索引，并按患者分组，只取每个人的第一个
        all_indices = (self.tokens == mm_id).nonzero(as_tuple=True)[0]
        self.anchor_indices = []
        seen_patients = set()
        
        for idx in all_indices:
            p_idx = self._get_patient_idx(idx.item())
            if p_idx not in seen_patients:
                self.anchor_indices.append(idx.item())
                seen_patients.add(p_idx)

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, idx):
        anchor_idx = self.anchor_indices[idx]
        patient_idx = self._get_patient_idx(anchor_idx)
        data_start_idx = int(self.patient_offsets[patient_idx])
        
        # 截断超长历史，保证 anchor_idx 之前的序列能喂进模型
        if anchor_idx - data_start_idx > self.timeline_len:
            data_start_idx = anchor_idx - self.timeline_len
            
        patient_context = self._get_patient_context(data_start_idx)
        # 输入序列：从历史开始到 ID 2034 确诊这一刻
        timeline = self.tokens[data_start_idx : anchor_idx + 1]
        
        x = th.cat((patient_context, timeline))
        y = {
            "expected": -1,
            "anchor_time": self.times[anchor_idx].item(), # 记录确诊日期
            "patient_id": patient_idx
        }
        return x, y
