# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedDPOTrainer():

    def __init__(self, dpo_engine, args):
        self.dpo_engine = dpo_engine
        self.actor_model = self.dpo_engine.actor
        # self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.dpo_engine.ref
        # self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.dpo_engine.tokenizer
        self.args = args
        # self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

    
    
    # @torch.jit.script
    def get_cumlog_probs(model, input_ids, attention_mask, labels, is_ref=False):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        labels: [batch_size, seq_len]

        Note that input_ids and labels should be of the same length, which implies that the 
        last token of input_ids should be the eos token.

        Returns:
            logprobs: [batch_size, seq_len] where logprobs[i, j] is the log probability of the
            whole sequence up to the jth token in the ith batch.
        """

        out = model(input_ids, attention_mask=attention_mask)
        # if is_ref:
        #     out = ref_model(input_ids, attention_mask=attention_mask)
        # out = torch.rand((input_ids.shape[0], input_ids.shape[1], 32000)) # CHANGE

        loss_mask = (labels == 2) # CHANGED FROM TOKENIZER.PAD_TOKEN_ID TO 2 FOR JIT COMPILATION

        logprobs = F.log_softmax(out.logits, dim=-1)
        # logprobs = F.log_softmax(out, dim=-1) # CHANGE
        per_token_logps = torch.gather(logprobs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        per_token_logps = per_token_logps.masked_fill(loss_mask, 0)

        per_token_logps = per_token_logps.cumsum(1)  # [batch_size, seq_len]

        # now every element in logprobs is the probability of the whole sequence up to that point
        return per_token_logps


    # @torch.jit.script
    def _get_loss(log_probs, ref_log_probs, simple_labels, 
             beta: torch.Tensor = torch.tensor(0.5), 
             label_smoothing: torch.Tensor = torch.tensor(0.03)):
    # In the github, authors of DPO paper have r = log_probs - ref_log_probs
    # In Scale paper, they say r = log(log_probs / ref_log_probs)
    # I am going with a riff on the implemented version

        r = log_probs / ref_log_probs
        # print(r)

        k = beta * r * simple_labels
        kp = -beta * r * simple_labels

        k[k == 0] = -torch.inf
        kp[kp == 0] = -torch.inf

        k = torch.sigmoid(k)
        kp = torch.sigmoid(kp)

        non_masked = torch.prod(k[k != 0])
        non_masked_p = torch.prod(kp[kp != 0])

        # With label_smoothing as shown in https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
        loss = (-torch.log(non_masked) / k[k != 0].numel()) * (1 - label_smoothing) - (torch.log(non_masked_p) / kp[kp != 0].numel() * label_smoothing)

        return loss, (beta * r * simple_labels).sum()


    def train_dpo(self, log_probs, ref_log_probs, simple_labels):
        # train the rlhf mode here
        ### process the old outputs
        # prompts = inputs['prompts']
        # log_probs = inputs['logprobs']
        # ref_log_probs = inputs['ref_logprobs']
        # reward_score = inputs['rewards']
        # values = inputs['value']
        # attention_mask = inputs['attention_mask']
        # simple_labels = inputs['simple_labels']
        # seq = inputs['input_ids']

        actor_loss, rewards = self._get_loss(log_probs, ref_log_probs, simple_labels)        
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, rewards

    
    def _validate_training_mode(self):
        assert self.actor_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.ref_model.module.training

    def train(self):
        self.actor_model.train()

    def eval(self):
        self.actor_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)

        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
   


class DeepSpeedDPOTrainerUnsupervised(DeepSpeedDPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
