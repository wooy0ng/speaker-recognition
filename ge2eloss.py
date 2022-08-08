import torch
import torch.nn as nn
import torch.nn.functional as F

class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([init_w]))
        self.b = nn.Parameter(torch.FloatTensor([init_b]))

        self.embed_loss = self.embed_loss_softmax

    def cosine_sim(self, dvector):
        n_speaker, n_utterance, d_embed = dvector.size()
        
        # embedding
        dvector_expns = dvector.unsqueeze(-1).expand(n_speaker, n_utterance, d_embed, n_speaker)
        dvector_expns = dvector_expns.transpose(2, 3)

        # centriods
        centroids = dvector.mean(dim=1).to(dvector.device)
        centroids = centroids.unsqueeze(0).expand(n_speaker*n_utterance, n_speaker, d_embed)
        centroids = centroids.reshape(-1, d_embed)

        # similarity
        dvector_rolls = torch.cat([dvector[:, 1:, :], dvector[:, :-1, :]], dim=1)
        dvector_excls = dvector_rolls.unfold(1, n_utterance-1, 1)
        mean_excls = dvector_excls.mean(dim=-1).reshape(-1, d_embed)

        indices = _indices_to_replace(n_speaker, n_utterance).to(dvector.device)
        ctrd_excls = centroids.index_copy(0, indices, mean_excls)
        ctrd_excls = ctrd_excls.view_as(dvector_expns)

        return F.cosine_similarity(dvector_expns, ctrd_excls, 3, 1e-6)

    def embed_loss_softmax(self, dvector, cos_sim_matrix):
        """Calculate the loss on each embedding by taking softmax."""
        n_speaker, n_utterance, _ = dvector.size()
        indices = _indices_to_replace(n_speaker, n_utterance).to(dvector.device)
        losses = -F.log_softmax(cos_sim_matrix, 2)
        return losses.flatten().index_select(0, indices).view(n_speaker, n_utterance)

    def forward(self, dvecs):
        cos_sim_matrix = self.cosine_sim(dvecs)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()

def _indices_to_replace(n_spkr, n_uttr):
    indices = [
        (s * n_uttr + u) * n_spkr + s for s in range(n_spkr) for u in range(n_uttr)
    ]
    return torch.LongTensor(indices)