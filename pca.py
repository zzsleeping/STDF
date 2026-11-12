"""Build the frozen PCA memory D = PCA_M(E^T)^T in R^{M x d} from GPT-2 word embeddings."""
import argparse

import torch
from sklearn.decomposition import PCA
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

parser = argparse.ArgumentParser()
parser.add_argument('--gpt_path', type=str, default='gpt2')
parser.add_argument('--components', type=int, default=500)
parser.add_argument('--output', type=str, default='wte_pca_500.pt')
args = parser.parse_args()

model = GPT2Model.from_pretrained(args.gpt_path)
wte = model.wte.state_dict()['weight'].cpu().float().numpy()  # [V, d]

pca = PCA(n_components=args.components, svd_solver='full')
memory = pca.fit_transform(wte.T).T  # [M, d]

torch.save(torch.from_numpy(memory).float().contiguous(), args.output)
print('saved {} with shape {}'.format(args.output, tuple(memory.shape)))
