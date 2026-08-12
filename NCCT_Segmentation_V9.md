### **System Context and Objective**

**Goal:** Implement a custom multi-objective loss function in PyTorch for a Generative Adversarial Network (GAN) called MAGIC (Multitask Automated Generation of Intermodal CT perfusion maps).

**Architecture Context:**
The network translates a single domain input (Non-Contrast CT, or $X$) into four distinct target perfusion maps ($CTP$) simultaneously.

- **$N$**: Number of perfusion maps, which is **4**.
- **$PM$**: A specific Perfusion Map type where $PM \in \{CBV, CBF, MTT, TTP\}$.
- **$x$**: The real input NCCT image.
- **$G_{PM}(x)$**: The generated synthetic perfusion map for a specific type.
- **$D_{PM}$**: The PatchGAN discriminator corresponding to a specific map type.

---

### **1. Total Objective Function**

The overall generator objective is a weighted sum of four independent loss components. The generator aims to minimize this objective, while the discriminator maximizes the GAN component:

$$
G^{*}=arg~min_{G_{CTP}}max_{D_{CTP}}\mathcal{L}_{GAN}(G_{CTP},D_{CTP},CTP,X)+\lambda_{1}\mathcal{L}_{L1}(G_{CTP},CTP,X)+\lambda_{2}\mathcal{L}_{EXT}(G_{CTP},CTP,X)+\lambda_{3}\mathcal{L}_{MML}(G_{CBF},G_{MTT},CBV,X)
$$

_Coding requirement:_ The loss class/function should accept three hyperparameters ($\lambda_{1}$, $\lambda_{2}$, $\lambda_{3}$) to weight the loss components.

---

### **2. Component Specifications**

#### **A. GAN Loss ($\mathcal{L}_{GAN}$)**

A modified Pix2Pix adversarial loss applied across all $N$ generated maps:

$$
\mathcal{L}_{GAN}(G_{CTP},D_{CTP},CTP,X)=\frac{1}{N}\sum_{PM\in CTP}\mathbb{E}_{PM\sim p_{data}(PM)}[log~D_{PM}(PM)]+\mathbb{E}_{x\sim p_{data}(x)}[log(1-D_{PM}(G_{PM}(x)))]
$$

_Coding requirement:_ Average the standard binary cross-entropy (BCE) adversarial loss across the 4 generated maps and their respective discriminators.

#### **B. Structural Fidelity / L1 Loss ($\mathcal{L}_{L1}$)**

Measures the L1 distance (Mean Absolute Error) between the real and synthesized maps to ensure structural consistency:

$$
\mathcal{L}_{L1}(G_{CTP},CTP,X)=\frac{1}{N}\sum_{PM\in CTP}\mathbb{E}_{(x,PM)\sim(p_{data}(x),P_{data}(PM))}[\vert{}\vert{}PM-G_{PM}(x)\vert{}\vert{}_{1}]
$$

_Coding requirement:_ Calculate the average L1 loss across all 4 map types.

#### **C. Multimodal Loss ($\mathcal{L}_{MML}$)**

A physiology-informed loss based on the Central Volume Principle ($CBV=CBF\times MTT$). It minimizes the L1 distance between the real $CBV$ map and the product of the synthesized $CBF$ and $MTT$ maps:

$$
\mathcal{L}_{MML}(G_{CBF},G_{MTT},CBV,X)=\mathbb{E}_{(x,CBV)\sim(p_{data}(x),p_{data}(CBV))}[\vert{}\vert{}G_{CBF}(x)\times G_{MTT}(x)-CBV\vert{}\vert{}_{1}]
$$

_Coding requirement:_ Perform an element-wise multiplication of the generated CBF and MTT tensors, then compute the L1 loss against the ground-truth CBV tensor.

#### **D. Extrema Loss ($\mathcal{L}_{EXT}$)**

A custom loss designed to heavily penalize errors in regions of interest (ROIs) that deviate significantly from the image mean (e.g., highly ischemic tissue). It uses an element-wise product of a weight map ($W_x$) and a squared error map ($H_x$).

**Step 1: Calculate the Weight Map ($W_x$)**
The real perfusion map is min-max normalized, shifted by 0.5 to a range of [-0.5, 0.5], and then squared:

$$
W_{x}=(\frac{PM-min(PM)}{max(PM)-min(PM)}-0.5)^{2}
$$

**Step 2: Calculate the Error Map ($H_x$)**
The mean squared error (MSE) between the min-max normalized generated map and the min-max normalized real map:

$$
H_{x}=(\frac{G_{PM}(x)-min(G_{PM}(x))}{max(G_{PM}(x))-min(G_{PM}(x))}-\frac{PM-min(PM)}{max(PM)-min(PM)})^{2}
$$

**Step 3: Calculate the Final Extrema Loss**
Average the element-wise multiplication ($\odot$) of $W_x$ and $H_x$ over all map types:

$$
\mathcal{L}_{EXT}(G_{CTP},CTP,X)=\frac{1}{N}\sum_{PM\in CTP}\mathbb{E}_{(x,PM)\sim(p_{data}(x),p_{data}(PM))}[W_{x}\odot H_{x}]
$$

_Coding requirement:_ Be careful with PyTorch tensor dimensions during min-max normalization. Ensure the `min` and `max` operations are calculated per image (or per batch item appropriately) to avoid zero-division errors by adding a small epsilon.

### **Reference Paper**

Khan, W., Rees, J., See, K. B., Kato, S., Huang, Z., Lazarte, A., Douglas, K., Lou, X., Peng, T. J., Rajderkar, D., Sanelli, P., Singh, A., Tuna, I., Wilson, C. A., & Fang, R. (2026). Diagnostically Competitive Performance of a Physiology-Informed Generative Multi-Task Network for Contrast-Free CT Perfusion. _arXiv preprint arXiv:2505.22673v2_.
