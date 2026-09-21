<!-- https://karpathy.github.io/2019/04/25/recipe/ -->
<!-- Changes
1. Removing self.alpha_bars = torch.cumprod(self.alphas, dim=0) from NoiseSchuler cause it already fine it increases the noise too much on early steps around 200-300

2. Add an MLP learnable layer in TimeEmbeddings it was written in original paper
3. Shift the order of Conv -> Norn -> SILU to Norm -> SILU -> Conv as the latest papers
4. Add one more activaion inside forward of Resblock in residual_conv
5. Increae num_groups to 32 from 2 . cause previously it had 64 out channels so that means 32 groups in some layer 128 channels so 64 groups but now i has 128 channels at starting means 64  groups and at max 512 channels means 256 groups too much osillation and variance 
6. Decrease the LR from 3e-4 to 1e-4 and also considering changing the lr scheuler to onecyclelr scduler
7. The loop in sampling goes from 999-1 it should be shift to 999-0 and also add a Denom zero protector
 -->


### Denoising Diffusion Probabilistic Model

Implemented from scratch using  "Denoising Diffusion Probabilistic Models" by Ho et al., implemented every component from the math — forward process, noise scheduler, UNet with time embeddings, reverse sampling — and trained it on two datasets.

