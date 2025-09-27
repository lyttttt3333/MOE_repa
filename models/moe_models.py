from torch import Tensor
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from models.moe_helper import _router_z_loss, _capacity, einsum
import torch
import torch.distributed as dist
import torch.nn.functional as F
# from einops import einsum
import os
import random
import numpy as np

# 设置PyTorch打印时显示所有值，不显示省略号
torch.set_printoptions(profile="full")

# Reproducibility


gumbel_map: Dict[torch.device, Callable] = {}
def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def reduce_video_tokens(x, h, w, t):
    B, H, W, T, D = x.shape
    x = x.view(
        B,
        H // h, h,
        W // w, w,
        T // t, t,
        D
    )
    indexing_tokens = x.mean(dim=(2, 4, 6))
    calculate_tokens = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    # calculate tokens shape: [B, H//h, W//w, T//t, h, w, t, D]
    calculate_tokens = calculate_tokens.reshape(B, -1, D)

    return indexing_tokens, calculate_tokens

def recover_video_tokens(x, h, w, t, H, W, T):
    B, S, D = x.shape
    x = x.view(
        B,
        H // h, W // w, T // t,
        h, w, t,
        D
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.reshape(B, H, W, T, D)
    return x

def topkgating(logits: Tensor,
               topk: int,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_z_loss: bool = False,
               group_limited_greedy: bool = False,
               n_group: int = 8,
               topk_group: int = 3,
               norm_topk_prob: bool = True,
               routed_scaling_factor: float = 1.0,
               scoring_func: str = "softmax",
               e_score_correction_bias=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # TODO: FIXME: add parallel group
    if use_z_loss:
        z_loss = _router_z_loss(logits=logits, importance=used_token)
    else:
        z_loss = torch.tensor(0.0, dtype=logits.dtype)#.cuda(non_blocking=True)

    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    if scoring_func == "softmax":  # True, only support softmax
        # logits.shape = torch.Size([20, 10])
        gates = F.softmax(logits, dim=1)
        # gates.shape = torch.Size([20, 10])
    elif scoring_func == "sigmoid":
        gates = logits.sigmoid()
    else:
        raise RuntimeError("unsurpport scoring func type")

    if e_score_correction_bias is not None:  # False
        gates_for_choice = gates.detach() + e_score_correction_bias
    else:
        gates_for_choice = gates

    if group_limited_greedy:  # False
        group_shape = list(gates.shape[:-1]) + [n_group,
                                                gates.shape[-1] // n_group]  # (num_tokens, n_group, experts_per_group)
        if e_score_correction_bias is None:
            group_scores = (
                gates_for_choice.reshape(group_shape).max(dim=-1).values
            )  # [n, n_group]
        else:
            group_scores = gates_for_choice.reshape(group_shape).topk(2, dim=-1)[0].sum(dim=-1)
        # group_scores: (num_tokens, n_group)
        group_idx = torch.topk(
            group_scores, topk_group, dim=-1, sorted=False
        )[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                group_shape
            )
            .reshape(list(gates.shape))
        )  # [n, e]
        gates_for_choice = gates_for_choice.masked_fill(~score_mask.bool(), 0.0)

    expert_capacity = topk * _capacity(gates,
                                       torch.tensor(capacity_factor),
                                       torch.tensor(min_capacity))

    num_experts = int(gates.shape[1])
    tokens_per_group = int(gates.shape[0])
    
    expert_gate, expert_index = torch.topk(logits_w_noise if noisy_gate_policy == 'RSample' else gates_for_choice, topk)
    # expert_gate shape: [num_tokens, topk]
    # expert_index shape: [num_tokens, topk]

    expert_mask = F.one_hot(expert_index, num_experts)
    # batch size, topk, num_experts

    if used_token is not None:
        expert_mask = expert_mask.reshape(tokens_per_group, -1)

        expert_mask = einsum("s,se->se", used_token, expert_mask)

        expert_mask = expert_mask.reshape(tokens_per_group, -1, num_experts)
    # [num_tokens, topk, num_experts]

    expert_mask_aux = expert_mask.max(dim=-2)[0]
    # [num_tokens, topk, num_experts] -> [num_tokens, num_experts]

    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)

    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    # [seq, num_experts] -> [num_experts]

    l_aux = num_experts ** 2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)

    if norm_topk_prob and topk > 1:
        gates_s = torch.clamp(torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1),
                              min=torch.finfo(gates.dtype).eps)

        router_probs = gates / gates_s
    else:
        router_probs = gates
    
    router_probs = router_probs * routed_scaling_factor

    expert_index = torch.transpose(expert_index, 0, 1)

    expert_index = expert_index.reshape(-1)

    expert_mask = expert_mask.transpose(0, 1).reshape(-1, num_experts).to(torch.int32)
    # Shape: [topk*tokens_per_group, num_experts].
    
    # 这里统计的就是每个expert被选中的次数(top1+top2)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    if not drop_tokens:
        new_capacity = torch.max(exp_counts.sum(dim=0))
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        expert_capacity = new_capacity

    # expert_mask shape: [topk*tokens_per_group, num_experts]
    # token_priority shape: [topk*tokens_per_group, num_experts]
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1

    # Shape: [num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape((topk, -1, num_experts))

    # Shape: [tokens_per_group, num_selected_experts, num_experts].
    token_priority = torch.transpose(token_priority, 0, 1)

    token_priority = torch.max(token_priority, dim=1)[0]
    # Shape: [tokens_per_group, num_experts].

    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    # Shape: [tokens_per_group, num_experts].

    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)

    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    # Shape: [tokens_per_group, num_experts, expert_capacity].

    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    # Shape: [tokens_per_group, num_experts, expert_capacity].

    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    # router_probs shape: [tokens_per_group, num_experts]
    # dispatch_mask shape: [tokens_per_group, num_experts, expert_capacity]
    # combine_weights shape: [tokens_per_group, num_experts, expert_capacity]

    exp_counts_capacity = torch.sum(dispatch_mask)

    if used_token is None:
        exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)
    else:
        exp_capacity_rate = exp_counts_capacity / (torch.sum(used_token) * topk + 1e-8)

    return [l_aux, z_loss, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts

class Experts(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):

        super(Experts, self).__init__()

        self.num_experts = num_experts
        self.hidden_size = hidden_size

        self.gate_proj = torch.nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size),
            requires_grad=True
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size),
            requires_grad=True
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size),
            requires_grad=True
        )

        def init_param(param):
            import math
            for i in range(param.shape[0]):
                torch.nn.init.kaiming_uniform_(param[i], a=math.sqrt(5), nonlinearity='leaky_relu')

        init_param(self.gate_proj)
        init_param(self.up_proj)
        init_param(self.down_proj)

        # np.save("./np_array/gate_proj.npy", self.gate_proj.cpu().detach().numpy())
        # np.save("./np_array/up_proj.npy", self.up_proj.cpu().detach().numpy())
        # np.save("./np_array/down_proj.npy", self.down_proj.cpu().detach().numpy())

        # tmptmp
        # with torch.no_grad():
        #     self.gate_proj.copy_(torch.from_numpy(np.load("./np_array/gate_proj.npy")).cuda())
        #     self.up_proj.copy_(torch.from_numpy(np.load("./np_array/up_proj.npy")).cuda())
        #     self.down_proj.copy_(torch.from_numpy(np.load("./np_array/down_proj.npy")).cuda())

    def forward(self, inputs):
        # inputs shape: [num_experts, capacity, hidden_size]
        # self.gate_proj shape: [num_experts, intermediate_size, hidden_size]
        # self.up_proj shape: [num_experts, intermediate_size, hidden_size]  
        # self.down_proj shape: [num_experts, hidden_size, intermediate_size]
        
        # 使用 einsum 进行批量矩阵乘法，避免循环
        # 计算 gate 投影: [num_experts, capacity, hidden_size] @ [num_experts, hidden_size, intermediate_size] 
        # -> [num_experts, capacity, intermediate_size]
        gate_output = torch.einsum('ech,ehd->ecd', inputs, self.gate_proj.transpose(1, 2))
        
        # 计算 up 投影: [num_experts, capacity, hidden_size] @ [num_experts, hidden_size, intermediate_size]
        # -> [num_experts, capacity, intermediate_size]
        up_output = torch.einsum('ech,ehd->ecd', inputs, self.up_proj.transpose(1, 2))
        
        # SwiGLU 激活: silu(gate) * up
        gate_output = torch.nn.functional.silu(gate_output)
        inter_output = gate_output * up_output  # [num_experts, capacity, intermediate_size]
        
        # 下投影: [num_experts, capacity, intermediate_size] @ [num_experts, intermediate_size, hidden_size]
        # -> [num_experts, capacity, hidden_size]
        output = torch.einsum('ecd,edh->ech', inter_output, self.down_proj.transpose(1, 2))
        
        return output

class TopKGate(torch.nn.Module):

    wg: torch.nn.Linear

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 topk: int,
                 capacity_factor: float,
                 min_capacity: int,
                 use_z_loss: bool) -> None:
        super().__init__()

        self.wg = torch.nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)

        self.topk = topk
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.use_z_loss = use_z_loss

        self.noisy_gate_policy = None
        self.drop_tokens = True
        self.group_limited_greedy = False
        self.n_group = None
        self.topk_group = None     # group_limited_greedy
        self.norm_topk_prob = True # group_limited_greedy
        self.routed_scaling_factor = 1.0
        self.scoring_func = "softmax"
        self.e_score_correction_bias = None

        # np.save("./np_array/wg.npy", self.wg.weight.cpu().detach().numpy())
        # with torch.no_grad():
        #     self.wg.weight.copy_(torch.from_numpy(np.load("./np_array/wg.npy")).cuda())

    def forward(self, input: torch.Tensor, used_token: torch.Tensor = None):

        # hardcode input.float() to ensure fp32
        input_fp32 = input.float()
        input_fp32 = input_fp32.reshape(-1, input_fp32.shape[-1])
        # with torch.amp.autocast('cuda', enabled=False):
        logits = self.wg(input_fp32)
        
        gate_output = topkgating(logits,
                                 self.topk, 
                                 self.capacity_factor, 
                                 self.min_capacity, 
                                 used_token, 
                                 self.noisy_gate_policy, 
                                 self.drop_tokens, 
                                 self.use_z_loss, 
                                 self.group_limited_greedy, 
                                 self.n_group, 
                                 self.topk_group, 
                                 self.norm_topk_prob, 
                                 self.routed_scaling_factor, 
                                 self.scoring_func, 
                                 self.e_score_correction_bias)
        
        return gate_output

class PTMHunYuanMoE(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, topk, capacity_factor, min_capacity, use_z_loss, core_shape=(1,1,1)):
        super().__init__()

        self.experts = Experts(num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.topk_gate = TopKGate(num_experts=num_experts, hidden_size=hidden_size, topk=topk, capacity_factor=capacity_factor, 
                                  min_capacity=min_capacity, use_z_loss=use_z_loss)

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.core_size = core_shape

    def forward(self, hidden_states, used_token=None):
        
        # [bs, seqlen, hidden_size] -> [seqlen, bs, hidden_size]
        batch_size, T, H, W, hidden_size = hidden_states.shape
        device = hidden_states.device

        

        seq_len = H * W * T
        core_len = (self.core_size[0] * self.core_size[1] * self.core_size[2])
        seq_len_indexing = H * W * T // core_len
        indexing_tokens, hidden_states = reduce_video_tokens(hidden_states, *self.core_size)
        indexing_tokens = indexing_tokens.reshape(batch_size, -1, hidden_size)

        
        if used_token is None:
            # mock used_token
            used_token = torch.ones(size=[seq_len_indexing, batch_size], dtype=torch.int64, device = device).reshape(-1)


        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        reshaped_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        indexing_tokens = indexing_tokens.permute(1, 0, 2).contiguous()
        reshape_indexing_tokens = indexing_tokens.reshape(-1, indexing_tokens.shape[-1])

        # l_aux就是balance loss
        (l_aux, z_loss, exp_capacity_rate), combine_weights, dispatch_mask, exp_counts = self.topk_gate(reshape_indexing_tokens, used_token)
        # dispatch_mask shape: [topk*tokens_per_group, num_experts, expert_capacity]

        combine_weights = combine_weights.unsqueeze(1).expand(-1, core_len, -1, -1).flatten(0, 1)
        dispatch_mask = dispatch_mask.unsqueeze(1).expand(-1, core_len, -1, -1).flatten(0, 1)

        dispatched_input = einsum("sec,sm->ecm",
                                  dispatch_mask.type_as(hidden_states),
                                  reshaped_hidden_states)

        expert_output = self.experts(dispatched_input)

        combined_output = einsum("sec,ecm->sm",
                                 combine_weights.type_as(hidden_states),
                                 expert_output)
        
        # [seqlen, bs, hidden_size] -> [bs, seqlen, hidden_size]
        output = combined_output.reshape(seq_len, batch_size, combined_output.shape[-1]).permute(1, 0, 2).contiguous()

        output = recover_video_tokens(output, *self.core_size, T, H, W)

        return output, l_aux, z_loss


if __name__ == "__main__":

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    seq_shape = (4, 10, 10)
    core_shape = (2, 5, 5)
    hidden_size = 3

    num_experts = 8
    hidden_size = 3
    intermediate_size = 4
    topk = 2
    capacity_factor = num_experts
    min_capacity = 2
    use_z_loss = True

    inputs = torch.randn(batch_size, *seq_shape, hidden_size)
    print(f"input size: {inputs.shape}")



    ptmMoE = PTMHunYuanMoE(num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,
                        topk=topk, capacity_factor=capacity_factor, min_capacity=min_capacity, use_z_loss=use_z_loss, core_shape=core_shape)
    output, l_loss, z_loss= ptmMoE(inputs)
    print(output.shape)