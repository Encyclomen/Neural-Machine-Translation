class Beam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, prob, prev_beam, f_done):
        pre_score = prob.new(prev_beam.scores)
        score = prob + pre_score.unsqueeze(-1).expand_as(prob)
        nbest_score, nbest_idx = score.view(-1).topk(self.beam_size, largest=False)
        beam_idx = nbest_idx / prob.size(1)
        token_idx = nbest_idx - beam_idx * prob.size(1)

        sentence_done_list, score_done_list, remain_list = [], [], []
        prev_candidates = prev_beam.candidates
        for b_score, b_idx, t_idx in zip(nbest_score, beam_idx, token_idx):
            candidate = prev_candidates[b_idx] + [t_idx]

            if f_done(candidate):
                sentence_done_list.append(candidate)
                score_done_list.append(b_score)
            else:
                remain_list.append(b_idx)
                self.candidates.append(candidate)
                self.scores.append(b_score)
        return sentence_done_list, score_done_list, remain_list
