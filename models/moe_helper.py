import torch

# from ..hunyuan_moe import MoeConfig

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

def grouped_gemm_is_available():
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0`."
    )

ops = grouped_gemm.ops if grouped_gemm_is_available() else None
gg = grouped_gemm

class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts, config, expert_group_name):
        super().__init__()
        # self.grad_acc_fuse = args.gradient_accumulation_fusion
        self.grad_acc_fuse = False
        self.config = config
        self.num_local_experts = num_local_experts
        assert_grouped_gemm_is_available()
        # assert (
        #     config.remove_mlp_bias == True
        # ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        # self.expert_parallel = config.moe_expert_parallel_size > 1
        # self.sequence_parallel = config.sequence_parallel
        # self.activation = config.activation_func
        # if config.activation_func == 'swiglu':
        #     self.activation_func = SwiGLU()
        # elif config.activation_func == 'relu':
        #     self.activation_func = F.relu
        # elif config.activation_func == 'gelu':
        #     self.bias_gelu_fusion = config.bias_gelu_fusion
        #     self.activation_func = F.gelu
        # # How many feature each rank holds for fc1 and fc2, respectively.
        # if config.expert_tensor_parallel:
        #     tp_size = mpu.get_tensor_model_parallel_world_size()
        # else:
        #     tp_size = 1
        #     self.sequence_parallel = False
        self.expert_parallel = False
        self.sequence_parallel = False
        # hardcode SwiGLU
        self.activation_func = SwiGLU()
        tp_size = 1


        fc1_output_size = config.intermediate_size
        # if config.activation_func in ['reglu', 'geglu', 'swiglu']:
        #     # Project to 4h. If using swiglu double the output width,
        #     # see https://arxiv.org/pdf/2002.05202.pdf
        #     fc1_output_size *= 2
        fc1_output_size *= 2
        fc1_output_size_per_partition = fc1_output_size // tp_size

        fc2_input_size = config.intermediate_size
        fc2_input_size_per_partition = fc2_input_size // tp_size

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        # TODO: 测试下 use_cpu_initialization
        # if config.use_cpu_initialization:
        if False:
            self.group_h_to_4h = Parameter(
                torch.empty(
                    self.num_local_experts * fc1_output_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            self.group_4h_to_h = Parameter(
                torch.empty(
                    self.num_local_experts * self.config.hidden_size,
                    fc2_input_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                self.group_h_to_4h,
                self.config.hidden_size,
                fc1_output_size * self.num_local_experts,
                fc1_output_size_per_partition,
                partition_dim=1,
                init_method=init_method,
                params_dtype=config.params_dtype,
            )
            _initialize_affine_weight_cpu(
                self.group_4h_to_h,
                fc2_input_size,
                self.num_local_experts * self.config.hidden_size,
                fc2_input_size_per_partition,
                partition_dim=0,
                init_method=output_layer_init_method,
                params_dtype=config.params_dtype,
            )
        else:
            self.group_h_to_4h = torch.nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    fc1_output_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    # dtype=torch.bfloat16,
                ),
                requires_grad=True
            )
            self.group_4h_to_h = torch.nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    device=torch.cuda.current_device(),
                    # dtype=torch.bfloat16,
                ),
                requires_grad=True
            )

            torch.nn.init.xavier_uniform_(self.group_h_to_4h)
            torch.nn.init.xavier_uniform_(self.group_4h_to_h)
            # _initialize_affine_weight_gpu(
            #     self.group_h_to_4h,
            #     init_method,
            #     partition_dim=1,
            #     expert_parallel=self.expert_parallel,
            # )
            # _initialize_affine_weight_gpu(
            #     self.group_4h_to_h,
            #     output_layer_init_method,
            #     partition_dim=0,
            #     expert_parallel=self.expert_parallel,
            # )
            # setattr(self.group_h_to_4h, 'allreduce', False)
            # setattr(self.group_h_to_4h, 'group_name', expert_group_name)
            # setattr(self.group_4h_to_h, 'allreduce', False)
            # setattr(self.group_4h_to_h, 'group_name', expert_group_name)

        # self.recompute_swiglu = config.cheap_checkpoint_activations and config.activation_func == 'swiglu'
        # self.tp_size = tp_size
        self.recompute_swiglu = False
        self.tp_size = 1

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        # TODO(youngyfang)
        permuted_local_hidden_states = permuted_local_hidden_states.to(torch.bfloat16)
        tokens_per_expert = tokens_per_expert.cpu()
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.group_h_to_4h.view(self.num_local_experts, -1, self.config.hidden_size).to(torch.bfloat16)
            w2 = self.group_4h_to_h.view(self.num_local_experts, self.config.hidden_size, -1).to(torch.bfloat16)
            if self.grad_acc_fuse:
                w1_main_grad = self.group_h_to_4h.main_grad.view(self.num_local_experts, -1, self.config.hidden_size)
                w2_main_grad = self.group_4h_to_h.main_grad.view(self.num_local_experts, self.config.hidden_size, -1)
                self.group_h_to_4h.grad_added_to_main_grad = True
                self.group_4h_to_h.grad_added_to_main_grad = True

            if len(permuted_local_hidden_states.shape) == 4:
                ep_size, num_local_experts, capacity, dim_size = permuted_local_hidden_states.shape
                num_tokens = num_local_experts * ep_size * capacity
            else:
                # DeepEP
                # num_tokens = sum(tokens_per_expert)
                num_tokens, dim_size = permuted_local_hidden_states.shape
                num_local_experts = len(tokens_per_expert)

            if self.sequence_parallel: # False
                permuted_local_hidden_states = mpu.gather_from_sequence_parallel_region(permuted_local_hidden_states, tensor_parallel_output_grad=True)
                tokens_per_expert = mpu.get_tensor_model_parallel_world_size() * tokens_per_expert

                permuted_local_hidden_states = permuted_local_hidden_states.reshape(-1, num_local_experts, capacity, dim_size).transpose(0, 1).reshape(-1, dim_size).contiguous()
            else:
                if len(permuted_local_hidden_states.shape) == 4:
                    permuted_local_hidden_states = (
                        permuted_local_hidden_states.transpose(0, 1).reshape(num_local_experts * ep_size * capacity, -1)
                        .contiguous()
                    )

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=True, main_grad_b=w1_main_grad if self.grad_acc_fuse else None, grad_acc_fuse=self.grad_acc_fuse
            )
            # with SwigluHook(self.recompute_swiglu) as cm1:
            intermediate_parallel = self.activation_func(fc1_output)

            # with RecomputeSwigluHook(self.recompute_swiglu, cm1.inputs) as cm:
            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=True, main_grad_b=w2_main_grad if self.grad_acc_fuse else None, grad_acc_fuse=self.grad_acc_fuse)
            # if self.sequence_parallel:
            #     # 保证 reduce-scatter 的时候第一维是 tokens，而不是 num_local_experts
            #     fc2_output = fc2_output.reshape(num_local_experts, -1, dim_size).transpose(0, 1).reshape(-1, dim_size).contiguous()
            #     fc2_output = mpu.reduce_scatter_to_sequence_parallel_region(fc2_output)
            #     # 和外面的 shape 对应
            #     fc2_output = fc2_output.reshape(-1, num_local_experts, dim_size).transpose(0, 1).contiguous()
            # elif self.tp_size > 1:
            #     fc2_output = mpu.reduce_from_tensor_model_parallel_region(fc2_output)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.group_h_to_4h.view(self.config.hidden_size, -1)
            w2 = self.group_4h_to_h.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output
    

@torch.compile
def swiglu(x):
    x1, x2 = x.chunk(2, dim=(x.ndim - 1))
    return x1 * torch.nn.functional.silu(x2)


def SwiGLU():
    return swiglu


class PTMHunYuanMLP(torch.nn.Module):
    def __init__(self, config, is_shared_mlp=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if is_shared_mlp:
            intermediate_size = config.intermediate_size * config.num_shared_expert
        else:
            intermediate_size = config.intermediate_size
        # intermediate_size * 2 to adapt swiglu
        fc1_output_size = intermediate_size * 2
        self.dense_h_to_4h = torch.nn.Linear(self.hidden_size, fc1_output_size, bias=False)
        self.dense_4h_to_h = torch.nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SwiGLU()

    def forward(self, x):
        output = self.dense_4h_to_h(self.act_fn(self.dense_h_to_4h(x)))
        # adapt return api: (output, output_bias)
        return output, None

def _router_z_loss(logits, importance=None, num_microbatches=1):
  """Loss that encourages router logits to remain small and improves stability.
  Args:
    logits: a tensor with shape [<batch_dims>, experts_dim]
    num_microbatches: number of microbatches
    importance: an optional tensor with shape [<batch_dims>]
  Returns:
    z_loss: scalar loss only applied by non-padded tokens and normalized by
      num_microbatches.
  """

  # logits.shape = torch.Size([20, 10])
  log_z = torch.logsumexp(logits, -1)
  # log_z.shape = torch.Size([20])

  pow_z = torch.pow(log_z, 2)
  # pow_z.shape = torch.Size([20])

  # importance = tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
  if importance == None:
      importance = torch.ones(logits.size()[0], dtype=logits.dtype, device=logits.device)
  mask_z = pow_z * importance.to(pow_z.dtype)
  # mask_z.shape = torch.Size([20])

  total_token = torch.sum(importance)
  # total_token = 12.0
  # num_microbatches = 1
  z_loss = torch.sum(mask_z)/(total_token * num_microbatches)

  return z_loss


@torch.jit.script
def _capacity(gates: torch.Tensor, capacity_factor: torch.Tensor, min_capacity: torch.Tensor) -> torch.Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # num_tokens = 20
    # num_experts = 10

    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    # num_tokens / num_experts = 2.
    # capacity_factor = 1.25
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    # capacity = 3

    # min_capacity = 2
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)